"""Gradio application for the frozen, validation-selected complaint classifier."""
import os
from pathlib import Path
import sys

# Text-only, offline inference. Set these before importing Transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from complaints.inference import ComplaintPredictor
from complaints.text import normalize_text

DISPLAY_NAMES = {
    "credit_card": "Credit card",
    "credit_reporting": "Credit reporting",
    "debt_collection": "Debt collection",
    "mortgages_and_loans": "Mortgages and loans",
    "retail_banking": "Retail banking",
}
EXAMPLES = [
    ["My credit report shows an account that I never opened. I disputed it but it is still listed."],
    ["A debt collector keeps calling about a debt that does not belong to me."],
    ["My mortgage servicer applied my payment incorrectly and charged a late fee."],
    ["My credit card was charged twice and the card provider has not resolved my dispute."],
    ["My bank account was frozen and I cannot access my deposited funds."],
]
CSS = """
.gradio-container { max-width: 1120px !important; margin: auto !important; }
#hero { padding: 18px 0 10px; }
#hero h1 { font-size: clamp(26px, 3.8vw, 38px); font-weight: 700; letter-spacing: -0.9px; margin-bottom: 6px; }
#hero p { font-size: 15px; color: var(--body-text-color-subdued); max-width: 820px; line-height: 1.6; }

#prediction-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    padding: 18px;
    background: var(--background-fill-secondary);
}

#headline h3 {
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 4px;
}

/* Eliminate the distracting dotted lines */
.confidence-set .line {
    display: none !important;
}

/* Remove redundant giant header inside Label component */
.output-class {
    display: none !important;
}

/* Modern typography and card spacing for each category */
.confidence-set {
    font-family: inherit !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 14px !important;
    width: 100% !important;
}

.confidence-set .inner-wrap {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
}

/* Category label on left, clean bold percentage on right */
.confidence-set .label {
    order: 1 !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    width: 100% !important;
    margin-bottom: 6px !important;
}

.confidence-set .text {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: var(--body-text-color) !important;
}

.confidence-set .confidence {
    font-size: 14px !important;
    font-weight: 700 !important;
    color: var(--primary-600, #2563eb) !important;
    margin-left: auto !important;
}

/* Sleek, modern progress bar placed underneath the text */
.confidence-set .bar {
    order: 2 !important;
    height: 7px !important;
    border-radius: 6px !important;
    background: linear-gradient(90deg, #2563eb, #06b6d4) !important;
    margin: 0 !important;
    align-self: flex-start !important;
}

#footer { font-size: 13px; opacity: 0.8; padding-top: 14px; }
"""


def create_app(model_dir=None):
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR", ROOT / "deployment_model"))
    if not (model_dir / "bundle.json").exists():
        raise FileNotFoundError("The trained deployment_model bundle is missing.")
    torch.set_num_threads(max(1, min(4, int(os.environ.get("APP_CPU_THREADS", "2")))))
    predictor = ComplaintPredictor(model_dir, device="cpu")

    def classify(text):
        try:
            scores = predictor.predict(text)
        except ValueError as exc:
            # Clear the previous prediction on invalid submissions.
            return "### Enter a valid complaint", {}, str(exc)
        winner = max(scores, key=scores.get)
        input_length, cap = predictor.input_length(text)
        status = "Prediction complete."
        if input_length > cap:
            status += f" This complaint exceeds {cap} tokens; the model used its opening portion."
        headline = f"### {DISPLAY_NAMES[winner]}\nModel probability: **{scores[winner]:.1%}**"
        return headline, {DISPLAY_NAMES[k]: v for k, v in scores.items()}, status

    model_name = {"distilbert": "DistilBERT", "rnn": "SimpleRNN", "lstm": "LSTM", "gru": "GRU"}[predictor.kind]
    summary = ""
    metrics_path = ROOT / "reports/final_test" / predictor.kind / "metrics.json"
    if metrics_path.exists():
        import json
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary = f"Recorded held-out results: **{metrics['accuracy']:.2%} accuracy**, **{metrics['macro_f1']:.4f} macro F1**. These describe the archived experiment.\n\n"
    cap = predictor.preprocessing["transformer_max_length" if predictor.kind == "distilbert" else "recurrent_max_length"]
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="teal", neutral_hue="slate")
    with gr.Blocks(title="Consumer Complaint Classifier", theme=theme, css=CSS) as app:
        gr.Markdown(
            "NLP TRAINING PROJECT · COMPLAINT CLASSIFICATION\n\n# Consumer Complaint Classifier\n"
            "A prototype supporting complaint categorization for financial products and services. "
            "The system analyzes customer narratives in real-time to predict the primary resolution category.",
            elem_id="hero",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                complaint = gr.Textbox(
                    lines=8, max_lines=15, label="Your complaint",
                    placeholder="For example: My bank froze my account and I cannot access my deposited funds…",
                    info="English text · up to 20,000 characters. Avoid entering private account details.",
                )
                with gr.Row():
                    submit = gr.Button("Classify complaint", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                status = gr.Textbox(label="Status", value="Ready for a complaint.", interactive=False)
            with gr.Column(scale=5, elem_id="prediction-card"):
                headline = gr.Markdown("### Your result will appear here")
                output = gr.Label(label="Probability across all categories", num_top_classes=5, show_heading=False)
        gr.Examples(examples=EXAMPLES, inputs=complaint, label="Try an example")
        with gr.Accordion("Categories and model limitations", open=False):
            gr.Markdown(
                "**Categories:** credit cards, credit reporting, debt collection, mortgages and loans, and retail banking.\n\n"
                f"Loaded model: **{model_name}**.\n\n"
                + summary +
                "Scores are uncalibrated model probabilities. The model always chooses among the five categories, "
                "including for unrelated text. It was trained on narratives already processed by the publisher, "
                f"so naturally written complaints may behave differently. Long complaints are truncated to {cap} tokens."
            )
        gr.Markdown("Consumer Complaint Classification · Training project prototype", elem_id="footer")
        outputs = [headline, output, status]
        submit.click(classify, inputs=complaint, outputs=outputs, api_name="classify", concurrency_limit=1)
        complaint.submit(classify, inputs=complaint, outputs=outputs, api_name=False, concurrency_limit=1)
        clear.click(lambda: ("", "### Your result will appear here", {}, "Ready for a complaint."),
                    outputs=[complaint, headline, output, status], api_name=False, queue=False)
    return app.queue(default_concurrency_limit=1, max_size=24)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    server_name = os.environ.get("SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1")
    create_app().launch(server_name=server_name, server_port=port,
                        show_error=False, share=False)
