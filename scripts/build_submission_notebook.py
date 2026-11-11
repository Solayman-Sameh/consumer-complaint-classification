"""Build a portable, student-facing notebook from frozen experiment evidence.

The embedded payload contains small reports, real training examples, the exact
vocabulary/tokenizer and source excerpts. It contains no model weights or full
corpus. Re-running its default cells never downloads data or retrains models.
"""
from pathlib import Path
import base64
import csv
import hashlib
import io
import json
import textwrap
import zlib

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks/consumer_complaint_classification.ipynb"
MODELS = ["rnn", "lstm", "gru", "distilbert"]


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_csv(relative):
    return list(csv.DictReader(io.StringIO((ROOT / relative).read_text(encoding="utf-8"))))


def main():
    data = {
        "audit": read_json("reports/preparation_reports/audit.json"),
        "config": read_json("configs/experiment.json"),
        "selection": read_json("reports/selection.json"),
        "environment": read_json("reports/environment.json"),
        "benchmark": read_json("reports/cpu_inference_benchmark.json"),
        "recovery": read_json("reports/presentation/data_recovery_verification.json"),
        "examples": read_csv("reports/presentation/training_examples.csv"),
        "length_histogram": read_csv("reports/presentation/training_length_histogram.csv"),
        "vocabulary": read_json("reports/presentation/training_vocabulary.json"),
        "preprocessing": read_json("deployment_model/preprocessing.json"),
        "tokenizer": (ROOT / "deployment_model/model/tokenizer.json").read_text(encoding="utf-8"),
        "models": {},
        "source": {name: (ROOT / "src/complaints" / name).read_text(encoding="utf-8")
                   for name in ["text.py", "models.py", "training.py", "preparation.py"]},
    }
    for model in MODELS:
        data["models"][model] = {
            "result": read_json(f"reports/models/{model}/result.json"),
            "history": read_json(f"reports/models/{model}/history.json"),
            "test": read_json(f"reports/final_test/{model}/metrics.json"),
            "confusion": read_csv(f"reports/final_test/{model}/confusion_matrix_counts.csv"),
            "errors": read_csv(f"reports/final_test/{model}/error_and_correct_examples.csv"),
        }
    for key, filename in [("app_verification", "reports/application_verification.json"),
                          ("deployment", "reports/deployment.json")]:
        data[key] = read_json(filename) if (ROOT / filename).exists() else {"status": "pending"}
    encoded = base64.b64encode(zlib.compress(json.dumps(data, ensure_ascii=False).encode(), 9)).decode()
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cells = []

    def md(s):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))

    def code(s, **metadata):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip(), metadata=metadata))

    md("""
    # Consumer Complaint Classification
    ### From financial narratives to complaint routing
    **Instant AI & Data Analysis Training · Sprint 03 · Project 01**

    We compare three custom recurrent classifiers—SimpleRNN, LSTM and GRU—with a fine-tuned DistilBERT.
    The task is to assign an English complaint to one of five financial-product categories.

    **Completed experiment:** 123,978 retained complaints; four trained models; a deployment model
    selected using validation macro F1 before the final test phase. DistilBERT achieved **85.43% test
    accuracy** and **0.8467 test macro F1**.

    **How to use this notebook:** read the saved outputs, or Run All to rebuild its analysis on CPU.
    The default walkthrough is self-contained: small evidence files, the trained vocabulary and
    tokenizer are embedded. No dataset download, model-weight download, GPU or retraining is triggered.
    Historical training results are explicitly loaded from the completed run. Section 12 offers
    optional live inference from an existing local bundle. The separate executed Kaggle notebook
    preserves the full download → preparation → training → evaluation → export workflow.

    **Contents:** 1. Evidence · 2. Dataset · 3. EDA · 4. Preprocessing · 5. Sequences and padding ·
    6. Embeddings · 7. Architectures · 8. Training · 9. Validation selection · 10. Final evaluation ·
    11. Error analysis · 12. Inference and application · 13. Conclusions and reproducibility.
    """)
    md("""
    ## 1. Load the frozen experiment evidence
    The compact cell below unpacks report data, not trained neural weights. Keeping the evidence
    inside the notebook makes tables and figures reproducible after moving this file.
    The original report files remain in the project’s `reports/` directory.

    Dependencies for Run All: Python 3.11+, NumPy, pandas, Matplotlib, tokenizers and PyTorch.
    No installation command runs automatically. Kaggle already supplies most of these libraries.
    """)
    code(f"""
    import base64, hashlib, json, zlib
    PAYLOAD = {encoded!r}
    E = json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
    assert hashlib.sha256(json.dumps(E, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == {digest!r}
    print('Embedded evidence verified. Experiment:', E['selection']['signature'][:16])
    """, tags=["hide-input"], jupyter={"source_hidden": True})
    code("""
    from pathlib import Path
    import io, math, sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown, Code

    NAMES = {'rnn': 'SimpleRNN', 'lstm': 'LSTM', 'gru': 'GRU', 'distilbert': 'DistilBERT'}
    LABELS = E['audit']['labels']
    PRETTY = {label: label.replace('_', ' ').capitalize() for label in LABELS}
    COLORS = ['#94a3b8', '#14b8a6', '#3b82f6', '#7c3aed']
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.titlesize': 12, 'font.size': 10})
    audit, config = E['audit'], E['config']
    def show():
        plt.tight_layout()
        plt.show()
        plt.close('all')
    display(pd.DataFrame([{'Stage': 'Source CSV', 'Records': audit['source_rows']},
                          {'Stage': 'Retained', 'Records': audit['retained_rows']},
                          *[{'Stage': k, 'Records': v} for k, v in audit['split_counts'].items()]]))
    """)
    md("""
    ## 2. Understand the dataset and target
    Source: [Consumer Complaints Dataset for NLP, version 1](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp).
    Its 162,421 rows contain `Unnamed: 0`, `product`, and `narrative`. We predict `product` from
    `narrative`; the index column is not a model feature. There are no official train/validation/test splits.

    **Upstream processing matters:** all supplied narratives already contain lowercase alphabetic words
    and whitespace. Original punctuation, numbers, casing, and original complaint identifiers are
    unavailable. We cannot claim to recover raw complaints or undo the publisher’s cleaning.

    The following are **real training examples** from the exact completed experiment. Their source IDs
    identify rows in this dataset, not original CFPB complaint identifiers. Ten examples and the vocabulary
    were recovered from an already-downloaded archive; record, vocabulary and encoded-array hashes match
    the completed experiment byte for byte.
    """)
    code("""
    examples = pd.DataFrame(E['examples'])
    display(examples[['source_id', 'product', 'split']].head(10))
    display(examples[['product', 'narrative']].assign(narrative=lambda x: x.narrative.str.slice(0, 260)))
    print('Source CSV SHA-256:', audit['source_csv_sha256'])
    print('Recovered teaching assets:', E['recovery']['status'])
    """)
    md("""
    ## 3. Explore quality, class balance, and sequence lengths
    Cleaning removes blank rows, contradictory duplicate labels and repeated narratives. Additional
    grouping keeps complaints that become identical after tokenization/truncation in the same split.
    This is stronger than simply splitting rows randomly, although semantic paraphrases may remain.
    """)
    code("""
    flow = pd.DataFrame(audit['exclusion_flow'])
    display(flow)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.barh(flow.stage.str.replace('_', ' '), flow.remaining, color='#3b82f6')
    ax.invert_yaxis(); ax.set_xlabel('Records remaining'); ax.set_title('Dataset exclusion audit')
    for i, row in flow.iterrows():
        ax.text(row.remaining + 1200, i, f'{row.remaining:,}', va='center', fontsize=9)
    ax.set_xlim(0, 187000)
    show()
    """)
    code("""
    counts = pd.DataFrame(audit['class_counts_by_split']).reindex(LABELS)[['train','validation','test']]
    display(counts.assign(total=counts.sum(axis=1)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts.rename(index=PRETTY).plot.barh(stacked=True, ax=axes[0], color=['#2563eb','#14b8a6','#f59e0b'])
    axes[0].set_title('Retained categories across shared splits'); axes[0].set_xlabel('Complaints')
    weights = pd.Series(audit['training_class_weights']).reindex(LABELS)
    axes[1].barh([PRETTY[k] for k in LABELS], weights, color='#7c3aed')
    axes[1].set_title('Class weights fitted on training only'); axes[1].set_xlabel('N / (K × class count)')
    show()
    print(f"Largest class: {counts.train.max() / counts.train.sum():.1%} of training data.")
    print('Split policy:', audit['split_policy'])
    display(pd.DataFrame([audit['duplicate_overlap_between_splits']]).T.rename(columns={0:'Cross-split overlaps'}))
    """)
    md("""
    Credit reporting dominates the dataset, so accuracy alone could conceal weak smaller-category
    performance. All four models use training-derived class-weighted cross-entropy. Validation and
    test distributions retain their observed proportions. **Macro F1** gives every class equal weight.

    The final split contains **99,182 training / 12,398 validation / 12,398 test** records.
    The learned vocabulary and class weights use training data only. Fixed pretrained tokenization and
    duplicate grouping are used to audit split isolation; test predictions never guide model selection.
    """)
    code("""
    histogram = pd.DataFrame(E['length_histogram'])
    for col in ['left','right','count']: histogram[col] = histogram[col].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.7))
    for ax, kind, color in zip(axes, ['recurrent','transformer'], ['#14b8a6','#3b82f6']):
        frame = histogram.loc[(histogram.representation == kind) & (histogram.left < 800)]
        ax.bar(frame.left, frame['count'], width=frame.right-frame.left, align='edge', color=color)
        ax.axvline(256, color='#ef4444', linestyle='--', label='256-token cap')
        ax.set_title(f'Training lengths: {kind} tokens'); ax.set_xlabel('Tokens (view below 800)')
        ax.set_ylabel('Complaints'); ax.legend()
    show()
    display(pd.DataFrame([{'Representation': k,
        'Training median tokens': audit['lengths']['train'][k+'_length_percentiles']['50'],
        'Training p95 tokens': audit['lengths']['train'][k+'_length_percentiles']['95'],
        'Training truncated': audit['lengths']['train'][k+'_truncation_fraction']}
        for k in ['recurrent','transformer']]))
    """)
    md("""
    A 256-token cap covers most complaints but truncates **3.66%** of recurrent training inputs and
    **6.08%** of transformer training inputs. Word tokens and transformer subwords are different units.
    Length and capacity choices are part of this fixed experiment; the test set is not used to retune them.
    """)
    md("""
    ## 4. Follow the two preprocessing paths
    Recurrent models use NFKC normalization, case folding, non-letter removal, and a fixed stop-word
    list. Negations such as `not`, `never`, and `without` are preserved. Lemmatization/stemming is optional
    in the assignment and was omitted.

    DistilBERT uses the supplied narrative after normalization with its pretrained tokenizer. We do not
    additionally remove stop words or punctuation from this branch. The source itself was already cleaned;
    this comparison therefore concerns complete modeling pipelines, not architecture alone.
    """)
    code("""
    # Execute the exact small preprocessing module embedded from the experiment.
    exec(E['source']['text.py'])
    stopwords = set(E['preprocessing']['stopwords'])
    sample = examples.iloc[0].narrative
    display(pd.DataFrame({'Stage': ['Dataset narrative','Normalized','Recurrent-cleaned'],
                          'Text': [sample, normalize_text(sample), clean_recurrent(sample, stopwords)]}))
    # This invented sentence illustrates operations absent from the already-cleaned source.
    illustrative_raw = 'My BANK charged $25 twice!!! I did NOT authorize this payment.'
    display(pd.DataFrame({'Illustrative input': [illustrative_raw],
                          'Recurrent output': [clean_recurrent(illustrative_raw, stopwords)]}))
    assert 'not' in clean_recurrent(illustrative_raw, stopwords).split()
    """)
    md("""
    ## 5. Convert words into IDs, truncate, and pad
    The actual recurrent vocabulary contains **30,000 entries**, including `<PAD>=0` and `<UNK>=1`.
    Words are ordered by training frequency with an alphabetical tie break. Unknown validation or future
    words receive ID 1. Encoded arrays are stored with length 256, and the collator trims/pads each batch
    to its longest actual sequence. Packed sequences prevent padding from changing the final hidden state.
    """)
    code("""
    vocabulary = E['vocabulary']  # exact training-only vocabulary, not a toy mapping
    words = clean_recurrent(sample, stopwords).split()
    ids = encode_recurrent(sample, vocabulary, stopwords, config['recurrent_max_length'])
    padded = np.pad(np.asarray(ids), (0, 256-len(ids)))
    display(pd.DataFrame({'Word': words[:18], 'Training vocabulary ID': ids[:18]}))
    print('Actual word count:', len(words), '| Used:', len(ids), '| Stored shape:', padded.shape)
    print('Padding starts:', padded[len(ids):len(ids)+12].tolist())
    print('Unseen-word ID:', vocabulary.get('zzzxxyyunseenword', UNK_ID))
    assert vocabulary['<PAD>'] == 0 and vocabulary['<UNK>'] == 1
    assert len(vocabulary) == 30000 and np.all(padded[len(ids):] == 0)
    """)
    code("""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_str(E['tokenizer'])  # actual exported tokenizer; offline
    transformer_example = tokenizer.encode(normalize_text(sample))
    display(pd.DataFrame({'Subword / special token': transformer_example.tokens[:24],
                          'Token ID': transformer_example.ids[:24]}))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(length=256, pad_id=0, pad_token='[PAD]')
    encoded = tokenizer.encode(normalize_text(sample))
    print('Transformer input IDs:', len(encoded.ids), '| Mask length:', len(encoded.attention_mask))
    print('Non-padding tokens:', sum(encoded.attention_mask))
    assert len(encoded.ids) == len(encoded.attention_mask) == 256
    """)
    md("""
    ## 6. Understand the learned embedding
    An embedding maps each word ID to a dense vector. Our custom models learn a **30,000 × 128**
    table during supervised training. The next cell demonstrates the same layer and tensor shapes with
    freshly initialized values; it does **not** load trained embedding weights or train a model.
    Pretrained GloVe/FastText vectors are not required for this assignment.
    """)
    code("""
    import torch
    from torch import nn
    torch.manual_seed(42)
    embedding_demo = nn.Embedding(len(vocabulary), 128, padding_idx=0)
    demonstration_batch = torch.tensor([padded[:32]], dtype=torch.long)
    with torch.inference_mode(): vectors = embedding_demo(demonstration_batch)
    print('ID batch:', tuple(demonstration_batch.shape), '→ Embeddings:', tuple(vectors.shape))
    print('Embedding parameters:', embedding_demo.weight.numel())
    assert torch.count_nonzero(embedding_demo.weight[0]) == 0
    del embedding_demo, vectors
    """)
    md("""
    ## 7. Build and explain all four architectures
    The custom classifiers share the same learned embedding, hidden size, dropout and classification
    head. Their recurrent cell differs. RNN updates one hidden state; LSTM adds a gated memory cell;
    GRU uses update/reset gates with fewer parameters than LSTM. These gates can help preserve useful
    information across longer sequences.

    | Model | Data flow | Initialization |
    |---|---|---|
    | SimpleRNN | IDs → embedding → packed RNN → final hidden state → dropout → linear head | Learned from scratch |
    | LSTM | IDs → embedding → packed LSTM → final hidden state → dropout → linear head | Learned from scratch |
    | GRU | IDs → embedding → packed GRU → final hidden state → dropout → linear head | Learned from scratch |
    | DistilBERT | Subword IDs + mask → pretrained encoder → sequence-classification head | All encoder and head parameters fine-tuned |

    The next cell shows the **actual model definitions** used by the experiment. Showing them does not
    download pretrained weights or start training.
    """)
    code("display(Code(E['source']['models.py'], language='python'))")
    md("""
    ## 8. Training procedure and observed learning curves
    All models use the same record-level splits, class weights, and seed 42. Recurrent models use Adam,
    learning rate 0.001, batch size 128, and an eight-epoch ceiling. DistilBERT uses AdamW, learning rate
    0.00002, batch size 16 with two-step accumulation, and three epochs. Hidden and embedding dimensions
    are 128; classifier dropout is 0.3.

    Training uses weighted cross-entropy, gradient clipping, CUDA mixed precision, and early stopping
    with patience 2. Checkpoints are selected by validation macro F1. Two T4 GPUs execute independent
    models; DistilBERT itself is not distributed across both GPUs.

    The following function is the actual training implementation, including initialization, batching,
    loss, backpropagation, evaluation and checkpoint saving. It is shown for study. **Historical metrics
    below come from saved reports; this walkthrough does not pretend to retrain the models.**
    """)
    code("""
    import ast
    training_source = E['source']['training.py']
    tree = ast.parse(training_source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_model')
    display(Code('\\n'.join(training_source.splitlines()[function.lineno-1:function.end_lineno]), language='python'))
    """, jupyter={"outputs_hidden": True})
    code("""
    training = pd.DataFrame([{'Model': NAMES[k], 'Best epoch': v['result']['best_epoch'],
        'Epochs run': v['result']['epochs_run'], 'Trainable parameters': v['result']['trainable_parameters'],
        'Training minutes': v['result']['training_seconds_including_initialization']/60,
        'Checkpoint MiB': v['result']['checkpoint_bytes']/2**20} for k, v in E['models'].items()])
    display(training.round(3))
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    for row, (kind, record) in enumerate(E['models'].items()):
        h = pd.DataFrame(record['history'])
        axes[row,0].plot(h.epoch, h.training_loss, marker='o', label='Training')
        axes[row,0].plot(h.epoch, h.validation_loss, marker='o', label='Validation')
        axes[row,0].set_title(NAMES[kind] + ' · weighted loss'); axes[row,0].set_ylabel('Loss')
        axes[row,1].plot(h.epoch, h.validation_macro_f1, marker='o', color=COLORS[row], label='Validation macro F1')
        axes[row,1].axvline(record['result']['best_epoch'], color='#64748b', linestyle='--', label='Selected epoch')
        axes[row,1].set_title(NAMES[kind] + ' · validation'); axes[row,1].set_ylabel('Macro F1')
        for ax in axes[row]:
            ax.set_xlabel('Epoch'); ax.set_xticks(h.epoch); ax.grid(alpha=.18); ax.legend(fontsize=8)
    show()
    """)
    md("""
    LSTM and GRU substantially improve on the SimpleRNN baseline in this run. DistilBERT obtains the
    strongest validation macro F1. Its training loss falls while validation loss does not fall monotonically;
    the selected checkpoint follows the declared F1 criterion, not the lowest validation loss.

    Training durations are read consistently from `reports/models/*/result.json`. Some original notebook
    timing outputs differ from the extracted reports; timings are descriptive observations, not benchmark
    rankings. Concurrent workers and initialization affect them. The model scores and checkpoint identities
    agree with the frozen selection.
    """)
    md("""
    ## 9. Select using validation before looking at final test scores
    The predeclared rule chooses the highest validation macro F1. Within **0.001 absolute F1**, its
    tie priority is SimpleRNN → GRU → LSTM → DistilBERT. DistilBERT wins outside that tolerance.
    The saved selection has status `frozen_before_test`; final test outcomes do not change the winner.
    """)
    code("""
    validation = pd.DataFrame([{'Model': NAMES[k],
        'Validation accuracy': v['result']['validation_accuracy'],
        'Validation macro F1': v['result']['validation_macro_f1']} for k,v in E['models'].items()])
    display(validation.round(4))
    fig, ax = plt.subplots(figsize=(8,3.6))
    bars = ax.bar(validation.Model, validation['Validation macro F1'], color=COLORS)
    ax.set_ylim(0,1); ax.set_ylabel('Macro F1'); ax.set_title('Deployment selection uses validation only')
    ax.bar_label(bars, fmt='%.4f', padding=4)
    show()
    assert E['selection']['selected_model'] == 'distilbert'
    assert E['selection']['test_outcomes_used_for_selection'] is False
    print('Selected:', NAMES[E['selection']['selected_model']])
    """)
    md("""
    ## 10. Evaluate the frozen models on the held-out test split
    All four best checkpoints were evaluated in the final reporting phase on **12,398** test records.
    The cells below rebuild visualizations from saved test metrics and confusion counts. They perform
    no new test inference or tuning.
    """)
    code("""
    fields = ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']
    comparison = pd.DataFrame([{'Model': NAMES[k], **{f:v['test'][f] for f in fields}}
                                for k,v in E['models'].items()]).set_index('Model')
    display(comparison.round(4))
    fig, ax = plt.subplots(figsize=(10,4))
    comparison[['accuracy','macro_precision','macro_recall','macro_f1']].plot.bar(ax=ax,
        color=['#64748b','#14b8a6','#3b82f6','#7c3aed'], rot=0)
    ax.set_ylim(0,1); ax.set_ylabel('Score'); ax.set_title('Final held-out comparison')
    ax.legend(['Accuracy','Macro precision','Macro recall','Macro F1'], ncol=2, fontsize=9)
    show()
    """)
    code("""
    fig, axes = plt.subplots(2,2,figsize=(12,10))
    short_labels = ['Credit card','Credit report','Debt collect.','Mortgages/loans','Retail banking']
    confusion_counts = {}
    for ax, (kind, record) in zip(axes.flat, E['models'].items()):
        cm = pd.DataFrame(record['confusion']).iloc[:,1:].astype(int).to_numpy()
        confusion_counts[kind] = cm
        norm = cm / cm.sum(axis=1,keepdims=True)
        ax.imshow(norm, cmap='Blues',vmin=0,vmax=1)
        ax.set_xticks(range(5),short_labels,rotation=35,ha='right',fontsize=8)
        ax.set_yticks(range(5),short_labels,fontsize=8)
        ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(NAMES[kind] + ' · recall by true class')
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{norm[i,j]:.2f}',ha='center',va='center',fontsize=9,
                        color='white' if norm[i,j]>.5 else '#0f172a')
        assert cm.sum() == 12398
    show()
    display(pd.DataFrame(confusion_counts['distilbert'],index=LABELS,columns=LABELS).rename_axis('True / predicted'))
    """)
    code("""
    per_class = pd.DataFrame(E['models']['distilbert']['test']['per_class']).T.loc[LABELS]
    display(per_class.round(4))
    fig, ax = plt.subplots(figsize=(9,3.7))
    per_class[['precision','recall','f1-score']].rename(index=PRETTY).plot.barh(ax=ax,
        color=['#14b8a6','#3b82f6','#7c3aed'])
    ax.set_xlim(0,1); ax.set_xlabel('Score'); ax.set_title('DistilBERT · per-class held-out performance')
    show()
    """)
    md("""
    DistilBERT leads this single run with **0.8467 macro F1**, compared with **0.8263 GRU**,
    **0.8231 LSTM**, and **0.7263 SimpleRNN**. Its improvement over GRU is about **2.04 percentage
    points of macro F1**. This is an observed comparison, not a claim of statistical superiority.
    Credit-card and debt-collection F1 remain weaker than the best-performing classes, leaving room
    for improved routing. The confusion matrices explain where aggregate scores hide mistakes.
    """)
    md("""
    ## 11. Inspect mistakes and limitations
    Error examples below are from the held-out reporting phase and are used only for explanation.
    They were not fed back into training or model selection. High confidence does not ensure correctness.
    """)
    code("""
    error_rows = pd.DataFrame(E['models']['distilbert']['errors'])
    incorrect = error_rows.loc[error_rows.correct.astype(str).str.lower().eq('false')].copy()
    display(incorrect[['product','predicted_label','confidence','narrative']].head(5).assign(
        narrative=lambda x:x.narrative.str.slice(0,360)))
    cm = confusion_counts['distilbert'].copy(); np.fill_diagonal(cm,0)
    true_idx,pred_idx = np.unravel_index(cm.argmax(),cm.shape)
    print(f'Largest off-diagonal count: {PRETTY[LABELS[true_idx]]} → '
          f'{PRETTY[LABELS[pred_idx]]}: {cm[true_idx,pred_idx]:,} complaints.')
    """)
    md("""
    Main limitations are upstream preprocessing of unknown scope, possible semantic near-duplicates,
    loss of later information in long complaints, one random seed, and an English five-category task.
    Natural user text differs from the already-processed training source. A forced-choice classifier
    also assigns unrelated text to a category. Its probabilities are **uncalibrated**, particularly with
    class weighting; they should not be presented as guaranteed probabilities of correctness.
    """)
    md("""
    ## 12. Classify new complaints and use the Gradio application
    These newly written examples exercise the exported model without changing it. Their saved predictions
    are produced by an actual clean-process CPU reload, separate from the historical test results.
    They are demonstrations, not an additional generalization benchmark.
    """)
    code("""
    verification = E['app_verification']
    demonstrations = verification.get('natural_examples',[])
    if demonstrations:
        display(pd.DataFrame(demonstrations)[['input','expected_category','predicted_category','probability']])
    else:
        print('Application verification evidence is pending; no predictions are fabricated.')
    display(pd.DataFrame([E['benchmark']]))
    print('Historical CPU timing is hardware-specific; it does not predict hosted latency.')
    print('Current app verification:', verification.get('status'))
    deployment = E['deployment']
    if deployment.get('url'):
        display(Markdown(f"**Try the Gradio application:** [{deployment['url']}]({deployment['url']})"))
        print('Hosting verification:', deployment.get('status'))
    else:
        print('No hosted URL has been verified yet.')
    """)
    code("""
    # Optional: use an existing local trained bundle. No download fallback is permitted.
    ENABLE_LIVE_INFERENCE = False
    if ENABLE_LIVE_INFERENCE:
        import os
        os.environ['USE_TF'] = '0'; os.environ['USE_FLAX'] = '0'; os.environ['HF_HUB_OFFLINE'] = '1'
        candidates = [Path.cwd(),Path.cwd().parent]
        project = next((p for p in candidates if (p/'deployment_model/bundle.json').exists()),None)
        if project is None:
            raise FileNotFoundError('Place this notebook in the project notebooks/ directory with the existing model bundle.')
        sys.path.insert(0,str(project/'src'))
        from complaints.inference import ComplaintPredictor
        predictor = ComplaintPredictor(project/'deployment_model',device='cpu')
        display(predictor.predict('A collector calls me every day about a debt that is not mine.'))
    else:
        print('Live inference disabled. All default report cells run without model weights.')
    """)
    md("""
    ## 13. Conclusions and reproducibility
    The experiment delivers the four required trained architectures, shared audited splits, explicit
    preprocessing and padding, learned custom embeddings, a genuinely fine-tuned transformer, required
    metrics, and a Gradio application. DistilBERT is the validation-selected deployment model; GRU offers
    a smaller recurrent alternative with lower measured test macro F1 in this run.

    Useful future experiments include multiple seeds, validation-based calibration, long-document
    strategies, and evaluation on naturally written complaints. These would be **new experiments**;
    they should not overwrite the frozen held-out assessment.

    **Full training reproduction:** use `consumer_complaint_training.ipynb` in Kaggle
    with Internet and GPU T4 ×2. It embeds the full source, retrieves the pinned source version,
    verifies fingerprints, prepares splits and encodings, trains the four models, freezes validation
    selection, evaluates test once per compatible experiment and exports the bundle. Saved outputs document the historical Kaggle run; maintained source changes have not undergone another full GPU training run.

    **Evidence map:** `reports/preparation_reports/` → quality and splits;
    `reports/models/*/` → training histories and validation;
    `reports/selection.json` → frozen selection;
    `reports/final_test/` → held-out evaluation;
    `reports/application_verification.json` → app checks;
    `reports/deployment.json` → actual hosting state.

    Assignment source: *INSTANT AI & DATA ANALYSIS TRAINING.pdf*, pages 17–19.
    The assignment requires Gradio; Hugging Face Spaces is the selected hosting provider.
    """)
    code("""
    display(pd.DataFrame([E['environment']['versions']]).T.rename(columns={0:'Recorded training version'}))
    print('Recorded training GPUs:', [g['name'] for g in E['environment']['gpus']])
    print('Source CSV SHA-256:', audit['source_csv_sha256'])
    print('Preparation identity:', E['recovery']['completed_preparation_id'])
    print('Experiment signature:', E['selection']['signature'])
    assert all(v['result']['signature'] == E['selection']['signature'] for v in E['models'].values())
    assert all(v['test']['records'] == 12398 for v in E['models'].values())
    print('Evidence consistency checks passed. No training was run in this walkthrough.')
    """)
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "title": "Consumer Complaint Classification — Student Walkthrough",
        "evidence_sha256": digest,
    })
    nbf.validate(notebook)
    nbf.write(notebook, DESTINATION)
    print(f"Built {DESTINATION.name}: {len(cells)} cells; {DESTINATION.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
