"""Regression checks for all four real bundle types and embedded release sources."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',GRADIO_ANALYTICS_ENABLED='False')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from app import create_app
from complaints.inference import ComplaintPredictor

def main():
    torch.set_num_threads(2)
    (ROOT/'.qa').mkdir(exist_ok=True)
    subprocess.check_call([sys.executable,str(ROOT/'scripts/sync_training_notebook.py'),'--check'])
    for kind in ['distilbert','rnn','lstm','gru']:
        with tempfile.TemporaryDirectory(dir=ROOT/'.qa',prefix='bundle-check-') as temporary:
            root=Path(temporary)
            if kind=='distilbert':
                root=ROOT/'deployment_model'
            else:
                for name in ['labels.json','preprocessing.json']:
                    shutil.copyfile(ROOT/'deployment_model'/name,root/name)
                shutil.copyfile(ROOT/'reports/presentation/training_vocabulary.json',root/'vocabulary.json')
                checkpoint=torch.load(ROOT/f'artifacts/checkpoints/{kind}/best.pt',map_location='cpu',weights_only=True)
                torch.save(checkpoint['model_state'],root/'model.pt')
                (root/'architecture.json').write_text(json.dumps({'kind':kind,'vocab_size':30000,'num_labels':5,'embedding_dim':128,'hidden_dim':128,'dropout':0.3}))
                manifest={'model':kind,'training_status':'trained','verification_only':False,'file_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}}
                (root/'bundle.json').write_text(json.dumps(manifest))
            app=create_app(root)
            classify=next(f.fn for f in app.fns.values() if f.api_name=='classify')
            heading,scores,status=classify('My bank froze my checking account and I cannot withdraw money.')
            assert len(scores)==5 and abs(sum(scores.values())-1)<1e-6
            assert 'Prediction complete' in status
            for text in ['', '!!!', '12345', 'a'*20001]:
                heading,scores,status=classify(text)
                assert not scores and 'Enter a valid complaint' in heading
            heading,scores,status=classify('My bank froze my account. '*100)
            assert scores and '256 tokens' in status
            print('Bundle interface checks passed:',kind,flush=True)
            app.close()
    print('All release regression checks passed.')

if __name__=='__main__': main()
