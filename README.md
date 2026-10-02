# Cellula Internship — Week 2

This repository contains the two parts of the week 2 task.

## Task 0 — Quantization research

Folder: [`Task0_Quantization`](Task0_Quantization)

A notebook that explains why large models like BERT and LLaMA are hard to run, compares common solutions
(knowledge distillation, pruning, LoRA and quantization) and then focuses on quantization:

- the absmax quantization formulas, written in LaTeX
- an example on a small array, worked step by step
- a BERT-sized weight matrix: 8-bit vs 4-bit error, error for 2–8 bits, and the effect of outliers (one scale per row)
- 8-bit quantization of a real DistilBERT model: size before and after, and a comparison of its predictions

## Task 1 — Toxic content classification app

Folder: [`Task1_Toxic_Classifier_App`](Task1_Toxic_Classifier_App)

A Streamlit app that checks a typed comment or an image for toxic content. Images are first described in words by BLIP-1,
then the text is classified by the BiLSTM trained in the week 1 task into six toxicity labels.
Every input and its result are saved automatically in a CSV database, which can be viewed in the app.

Setup and usage instructions are in the app's [README](Task1_Toxic_Classifier_App/README.md).

## Repository structure

```text
Cellula_2week/
├── Task0_Quantization/
│   └── Quantization.ipynb
└── Task1_Toxic_Classifier_App/
    ├── models/
    │   ├── lstm_model.keras
    │   ├── tokenizer.json
    │   └── lstm_thresholds.csv
    ├── app.py
    ├── classifier.py
    ├── imagecaption.py
    ├── database.py
    ├── database.csv
    ├── requirements.txt
    └── README.md
```
