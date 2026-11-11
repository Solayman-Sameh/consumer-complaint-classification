"""Verify the public Hugging Face model against the local inference bundle."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache-dir',type=Path,help='Optional Hugging Face cache location.')
    args=parser.parse_args()
    if args.cache_dir: os.environ['HF_HUB_CACHE']=str(args.cache_dir.resolve())
    os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='0',HF_HUB_DISABLE_XET='1')
    from huggingface_hub import HfApi, snapshot_download
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'src'))
    from complaints.inference import ComplaintPredictor
    repo='Solayman-Sameh/consumer-complaint-classifier'
    info=HfApi().model_info(repo,files_metadata=True)
    weight=next(f for f in info.siblings if f.rfilename=='model.safetensors')
    local=root/'deployment_model/model/model.safetensors'
    with local.open('rb') as stream: local_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    assert weight.lfs.sha256==local_hash, 'Published weights differ from local weights'
    snapshot=Path(snapshot_download(repo,revision=info.sha,
        allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt'], max_workers=1))
    # Reuse identical verified bytes, avoiding a redundant 268 MB transfer.
    destination=snapshot/'model.safetensors'
    if not destination.exists(): os.link(local,destination)
    model=AutoModelForSequenceClassification.from_pretrained(repo).eval()
    tokenizer=AutoTokenizer.from_pretrained(repo)
    predictor=ComplaintPredictor(root/'deployment_model')
    assert model.config.id2label==predictor.model.config.id2label
    torch.set_num_threads(2)
    fixture=json.loads((root/'reports/validation_reload_fixture.json').read_text())
    difference=0.0
    for item in fixture:
        text=item['narrative']
        batch=tokenizer(text,truncation=True,max_length=256,return_tensors='pt')
        with torch.inference_mode(): scores=model(**batch).logits.softmax(-1)[0].numpy()
        expected=predictor.predict_probabilities(text)
        np.testing.assert_allclose(scores,expected,atol=2e-5,rtol=2e-4)
        difference=max(difference,float(np.max(np.abs(scores-expected))))
    report={'status':'passed','verified_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'repo_id':repo,'revision':info.sha,'load_call':'AutoModelForSequenceClassification.from_pretrained("'+repo+'")','weights_sha256':local_hash,'labels_match_local_bundle':True,'compared_examples':len(fixture),'max_probability_difference':difference,'weights_transfer':'Reused local bytes after comparison with published LFS SHA-256; repository metadata, config and tokenizer fetched online.','publication_performed':False}
    (root/'reports/huggingface_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
