# 📝 Text Summarization System using Pegasus Transformer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/License-MIT-orange)

### End-to-End NLP Pipeline for Abstractive Text Summarization

</div>
</div># # workflow 

1. update config.yml # everything stored in this file,i haven't got to every file to change  
2. update params.yml # used to update a paremeter, when we do model training configurations
3. update entity.py # define the return type of a functions
4. update the configuration manager in src config
5. update components
6. update pipeline 
7. update main.py
8. update the app.py
</div>

---

# 📖 Overview

Text Summarization is one of the most important Natural Language Processing (NLP) tasks. The objective is to generate a concise and meaningful summary while preserving the key information from the original text.

This project implements an end-to-end Text Summarization pipeline using Google's **PEGASUS Transformer Model**, fine-tuned on the **SAMSum Dataset**. The solution includes:

* Data Ingestion
* Data Validation
* Data Transformation
* Model Training
* Model Evaluation
* Prediction Pipeline
* FastAPI Deployment

The project follows a modular Machine Learning engineering architecture suitable for production environments.

---

# 🎯 Objectives

* Generate high-quality abstractive summaries.
* Fine-tune PEGASUS on dialogue datasets.
* Build reusable ML pipelines.
* Deploy the model using FastAPI.
* Evaluate performance using ROUGE metrics.

---

# 🏗️ Project Architecture

```text
                    ┌──────────────────┐
                    │  SAMSum Dataset  │
                    └─────────┬────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Data Ingestion   │
                    └─────────┬────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Data Validation  │
                    └─────────┬────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Data Transformation │
                    └─────────┬────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Pegasus Training │
                    └─────────┬────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Model Evaluation │
                    └─────────┬────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Prediction API   │
                    └─────────┬────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ FastAPI Service  │
                    └──────────────────┘
```

---

# 📂 Project Structure

```bash
Text_Summerization/
│
├── artifacts/
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   └── model_trainer/
│
├── config/
│   └── config.yaml
│
├── research/
│   ├── textsummarization.ipynb
│   └── s04_data_model.ipynb
│
├── logs/
│
├── src/
│   └── TextSummarizer/
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       │
│       ├── pipeline/
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_validation.py
│       │   ├── stage_03_data_transformation.py
│       │   ├── stage_04_model_trainer.py
│       │   ├── stage_05_model_evaluation.py
│       │   └── prediction.py
│       │
│       ├── utils/
│       ├── config/
│       ├── entity/
│       └── constants/
│
├── app.py
├── main.py
├── params.yaml
├── requirements.txt
├── setup.py
└── README.md
```

---

# 🧠 Model Information

## Base Model

```python
google/pegasus-cnn_dailymail
```

## Dataset

```python
SAMSum Dataset
```

### Dataset Example

```text
Amanda: Are we meeting today?
John: Yes, at 5 PM.
Amanda: Great, see you there.
```

### Summary

```text
Amanda and John will meet at 5 PM.
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Aakash556-AI/Text_Summerization.git

cd Text_Summerization
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training Pipeline

Run complete training workflow:

```bash
python main.py
```

This executes:

```text
Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
```

---

# 📊 Evaluation Metrics

The model is evaluated using:

* ROUGE-1
* ROUGE-2
* ROUGE-L
* ROUGE-Lsum

Example:

```python
{
    "rouge1": 0.42,
    "rouge2": 0.21,
    "rougeL": 0.39,
    "rougeLsum": 0.40
}
```

---

# 🌐 FastAPI Deployment

Start API Server:

```bash
python app.py
```

or

```bash
uvicorn app:app --reload
```

---

## Swagger UI

```text
http://localhost:8000/docs
```

---

# 📡 API Endpoints

## Home

### Request

```http
GET /
```

### Response

Redirects to Swagger Documentation.

---

## Train Model

### Request

```http
GET /train
```

### Response

```json
{
  "message": "Training successful!"
}
```

---

## Generate Summary

### Request

```http
POST /predict
```

### Body

```json
{
  "text": "Artificial Intelligence is transforming industries worldwide by enabling machines to learn from data and make intelligent decisions."
}
```

### Response

```json
{
  "input_text": "Artificial Intelligence is transforming industries worldwide by enabling machines to learn from data and make intelligent decisions.",
  "summary": "AI is transforming industries through intelligent decision-making."
}
```

---

# 🔍 Prediction Pipeline

Example:

```python
from TextSummarizer.pipeline.prediction import PredictionPipeline

predictor = PredictionPipeline()

text = """
Artificial Intelligence is transforming industries worldwide
by enabling machines to learn from data.
"""

summary = predictor.predict(text)

print(summary)
```

---

# 📦 Requirements

Main Libraries:

```text
transformers
datasets
torch
fastapi
uvicorn
pydantic
numpy
pandas
pyyaml
rouge-score
```

Install:

```bash
pip install -r requirements.txt
```

---

# 🛡️ Future Improvements

* Docker Deployment
* Streamlit Dashboard
* Batch Summarization
* PDF Summarization
* Multi-language Summarization
* AWS Deployment
* Kubernetes Deployment
* CI/CD Integration

---

# 📈 Results

| Metric     | Score     |
| ---------- | --------- |
| ROUGE-1    | Evaluated |
| ROUGE-2    | Evaluated |
| ROUGE-L    | Evaluated |
| ROUGE-Lsum | Evaluated |

---

# 👨‍💻 Author

### Akash Kumar Sinha

**B.Tech CSE (AI & ML)**
Guru Jambheshwar University of Science and Technology, Hisar

### Connect With Me

GitHub:

```text
https://github.com/Aakash556-AI
```

LinkedIn:

```text
https://www.linkedin.com/in/
```

---

# ⭐ If you like this project

Give this repository a star ⭐ and support the project.

```bash
⭐ Star this repository
🍴 Fork this repository
📢 Share with others
```





