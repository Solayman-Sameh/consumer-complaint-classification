"""Compare CPU inference on identical texts; never train or alter checkpoints."""
import csv
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time
os.environ.update(USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from complaints.inference import ComplaintPredictor
from complaints.models import RecurrentClassifier
from complaints.text import encode_recurrent

def main():
    torch.set_num_threads(2)
    pre=json.loads((ROOT/'deployment_model/preprocessing.json').read_text())
    vocab=json.loads((ROOT/'reports/presentation/training_vocabulary.json').read_text())
    texts={
        'short':['My bank froze my checking account and I cannot access my funds.', 'A debt collector is calling about a debt that is not mine.', 'My credit card has an unauthorized charge.'],
        'long':['My bank froze my checking account and I cannot access my funds. '*70, 'A debt collector is calling about a debt that is not mine. '*70, 'My credit card has an unauthorized charge. '*70],
    }
    results={}
    for kind in ['rnn','lstm','gru','distilbert']:
        if kind=='distilbert':
            predictor=ComplaintPredictor(ROOT/'deployment_model')
            predict=predictor.predict_probabilities
        else:
            checkpoint=torch.load(ROOT/f'artifacts/checkpoints/{kind}/best.pt',map_location='cpu',weights_only=True)
            model=RecurrentClassifier(kind,len(vocab),5).eval()
            model.load_state_dict(checkpoint['model_state'])
            def predict(text):
                ids=encode_recurrent(text,vocab,set(pre['stopwords']),pre['recurrent_max_length'])
                with torch.inference_mode():
                    return model(torch.tensor([ids]),torch.tensor([len(ids)])).softmax(-1)[0].numpy()
        results[kind]={}
        for length,examples in texts.items():
            for text in examples: predict(text)
            timings=[]
            for i in range(30):
                start=time.perf_counter(); scores=predict(examples[i%len(examples)]); timings.append((time.perf_counter()-start)*1000)
                assert np.isfinite(scores).all() and np.isclose(scores.sum(),1,atol=1e-6)
            results[kind][length]={'median_ms':float(np.median(timings)), 'p95_ms':float(np.percentile(timings,95))}
        print(kind, results[kind],flush=True)
    report={'measured_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'device':'cpu','threads':2,'batch_size':1,'measured_calls_per_stratum':30,'warmup_calls_per_stratum':3,'includes_preprocessing_tokenization_forward_softmax':True,'excludes_model_loading_and_HTTP':True,'texts':texts,'models':results,'versions':{'torch':torch.__version__},'platform':__import__('platform').platform(),'processor':__import__('platform').processor(),'limitation':'One local machine, synthetic complaints, one run; not a hosted service guarantee.'}
    (ROOT/'reports/cpu_model_comparison.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
