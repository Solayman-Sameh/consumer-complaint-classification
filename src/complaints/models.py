"""Comparable single-layer recurrent classifiers and genuine DistilBERT tuning."""
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class RecurrentClassifier(nn.Module):
    def __init__(self, kind, vocab_size, num_labels, embedding_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        if kind not in {"rnn", "lstm", "gru"}:
            raise ValueError(kind)
        self.architecture = dict(kind=kind, vocab_size=vocab_size, num_labels=num_labels,
                                 embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.recurrent = cls(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.recurrent(packed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.classifier(self.dropout(hidden[-1]))


def create_model(kind, data, config, pretrained=True):
    if kind != "distilbert":
        return RecurrentClassifier(kind=kind, vocab_size=len(data.vocabulary), num_labels=len(data.label_names),
                                   embedding_dim=config["embedding_dim"], hidden_dim=config["hidden_dim"], dropout=config["dropout"])
    from transformers import AutoConfig, AutoModelForSequenceClassification
    arguments = dict(num_labels=len(data.label_names), id2label={i: name for i, name in enumerate(data.label_names)},
                     label2id=data.labels["label2id"])
    if pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(
            config["transformer_checkpoint"], revision=config["transformer_revision"],
            use_safetensors=True, attn_implementation="eager", **arguments)
    else:
        hf_config = AutoConfig.from_pretrained(data.root / "tokenizer", local_files_only=True, **arguments)
        model = AutoModelForSequenceClassification.from_config(hf_config, attn_implementation="eager")
    assert all(parameter.requires_grad for parameter in model.parameters()), "Encoder must be fine-tuned"
    return model


def forward_batch(model, kind, batch, device):
    ids = batch["input_ids"].to(device, non_blocking=True)
    if kind == "distilbert":
        mask = (ids != 0).long()
        return model(input_ids=ids, attention_mask=mask).logits
    return model(ids, batch["lengths"])
