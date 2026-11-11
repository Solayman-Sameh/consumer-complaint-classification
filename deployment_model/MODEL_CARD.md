---
language:
- en
license: mit
tags:
- text-classification
- distilbert
- pytorch
- financial-services
- complaint-classification
metrics:
- accuracy
- f1
datasets:
- shashwatwork/consume-complaints-dataset-fo-nlp
pipeline_tag: text-classification
widget:
- text: "I have noticed unauthorized charges on my credit card that I never made."
- text: "Equifax has incorrect delinquency marks on my credit report that do not belong to me."
- text: "Debt collectors keep calling my workplace harassing me about a debt that was discharged."
- text: "My mortgage servicer misapplied my monthly escrow payment and charged illegal late fees."
- text: "The bank locked my checking account and I cannot access my direct deposit salary."
---

# 💳 Consumer Complaint Classification — DistilBERT

This is the fine-tuned **DistilBERT** model for automated multi-class classification of consumer financial complaints across five resolution categories:
1. `credit_card` (Credit card or prepaid card)
2. `credit_reporting` (Credit reporting, credit repair services, or other personal consumer reports)
3. `debt_collection` (Debt collection)
4. `mortgages_and_loans` (Mortgage, payday loan, title loan, or personal loan)
5. `retail_banking` (Checking or savings account, money transfers, or virtual currency)

## Performance
- **Test Accuracy:** 85.43%
- **Test Macro F1:** 0.8467
- **Test Loss:** 0.4215
- **CPU latency:** hardware- and input-dependent; see the repository CPU benchmark report. No service guarantee is claimed.

## Limitations
The source contains publisher-processed English text; raw narratives are unavailable. Scores are uncalibrated. Inputs are truncated to 256 tokens and always classified into five categories, including unrelated text. Dataset performance does not establish production routing accuracy.

## Repository
Complete code, Gradio web app, training reproduction, and evaluation notebooks are available at:
[GitHub: Solayman-Sameh/consumer-complaint-classification](https://github.com/Solayman-Sameh/consumer-complaint-classification)
