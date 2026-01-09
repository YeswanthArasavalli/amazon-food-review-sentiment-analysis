# Amazon Food Review Sentiment Analysis

## 📌 Project Overview

This project performs **sentiment analysis** on 568K+ Amazon Fine Food reviews using both classical NLP and deep learning approaches. It demonstrates end-to-end machine learning workflow: from data preprocessing, baseline modeling, transformer fine-tuning, to deployment on Hugging Face Spaces.

**Live Demo:** [Try it on Hugging Face Spaces](https://huggingface.co/spaces/YeswanthArasavalli/Amazon_Food_Review_Analysis)

---

## 🎯 Key Features

- **Data Processing:** Full preprocessing pipeline with text cleaning and tokenization
- **Baseline Model:** TF-IDF with Logistic Regression (~89% accuracy)
- **Deep Learning:** Fine-tuned DistilBERT Transformer (~94% accuracy)
- **Evaluation:** Comprehensive metrics including Confusion Matrix, ROC-AUC Curve, and error analysis
- **Deployment:** Interactive Gradio UI hosted on Hugging Face Spaces
- **Interpretability:** Misclassified examples analysis for model robustness

---

## 📊 Dataset

| Property | Details |
|----------|----------|
| **Source** | Amazon Fine Food Reviews (Kaggle) |
| **Records** | 568,454 reviews |
| **Sentiment Classes** | 3 (Negative, Neutral, Positive) |
| **Class Distribution** | 1-2: Negative, 3: Neutral, 4-5: Positive |

### Original Features
- Product metadata
- Timestamps
- Review text
- User ratings
- Helpfulness scores

---

## 🧠 Methodology

### 1. Baseline Model

| Metric | Value |
|--------|-------|
| **Technique** | TF-IDF (50k features) + Logistic Regression |
| **Accuracy** | ~89% |
| **F1-Score** | ~0.88 |

**Purpose:** Established performance benchmark before deploying deep learning models.

### 2. Fine-Tuned Transformer Model

| Parameter | Value |
|-----------|-------|
| **Model** | DistilBERT (uncased base) |
| **Architecture** | Transformer Encoder |
| **Training Environment** | Google Colab GPU |
| **Epochs** | 2–3 |
| **Batch Size** | 16 |
| **Accuracy** | ~94% |
| **F1-Score** | ~0.94 |

**Improvement:** DistilBERT significantly outperformed classical baseline while maintaining computational efficiency.

---

## 🧪 Evaluation & Insights

✅ **Confusion Matrix** – Identified misclassification patterns  
✅ **ROC-AUC Curve** – Validated model discrimination ability  
✅ **Class Distribution Check** – Ensured balanced predictions  
✅ **Error Analysis** – Examined misclassified examples for robustness  

These evaluations ensured the model wasn't just accurate but **robust and interpretable**.

---

## 🚀 Deployment

The trained DistilBERT model is deployed using:

- **Frontend:** Gradio UI for user-friendly interaction
- **Hosting:** Hugging Face Spaces (serverless deployment)
- **Functionality:** Users enter any review text and receive instant sentiment prediction with confidence scores

### Example Usage

**Input:**
```
"The pasta tasted awful and was completely dry."
```

**Output:**
```
Sentiment: Negative
Confidence: 0.94
```

---

## 🛠️ Tech Stack

| Component | Tools |
|-----------|-------|
| **Languages** | Python |
| **Deep Learning** | Transformers (Hugging Face), PyTorch |
| **Data Processing** | Scikit-learn, Pandas, NumPy |
| **Deployment** | Gradio, Hugging Face Spaces |
| **Environment** | Google Colab (GPU) |

---

## 📁 Project Structure

````
Amazon-Food-Review-Sentiment-Analysis/
│
├── data/                      # empty with instructions
│   ├── raw/                   # Original dataset from Kaggle
│   └── processed/             # Cleaned & preprocessed data
│
├── models/                    # empty or contains placeholder file
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Dataset loading utilities
│   ├── preprocessing.py       # Text cleaning & tokenization
│   ├── train_baseline.py      # TF-IDF + Logistic Regression
│   ├── train_bert.py          # BERT fine-tuning pipeline
│   └── inference.py           # Model inference/prediction
│
├── app.py                     # Gradio/Streamlit application
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore patterns
└── README.md                  # This file
````````

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda
- CUDA (optional, for GPU acceleration)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YeswanthArasavalli/Amazon-Food-Review-Sentiment-Analysis.git
   cd Amazon-Food-Review-Sentiment-Analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset:**
   - Download from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
   - Place in `data/raw/`

4. **Run preprocessing:**
   ```bash
   python src/preprocessing.py
   ```

5. **Train models:**
   ```bash
   python src/train_baseline.py
   python src/train_transformer.py
   ```

6. **Evaluate results:**
   ```bash
   python src/evaluate.py
   ```

---

## 📊 Results Summary

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|----------------|
| Baseline (TF-IDF + LR) | 89% | 0.88 | ~5 min |
| DistilBERT | 94% | 0.94 | ~45 min (GPU) |

---

## 🎓 Key Learnings

- Deep learning models significantly outperform classical NLP approaches for sentiment analysis
- Transfer learning with pre-trained models (DistilBERT) provides strong performance with lower computational cost
- Comprehensive evaluation beyond accuracy (confusion matrices, ROC curves) is crucial for production models
- Cloud environments like Google Colab make advanced ML accessible without expensive hardware

---

## 📞 Contact & Support

- **Author:** Yeswanth Arasavalli
- **GitHub:** [YeswanthArasavalli](https://github.com/YeswanthArasavalli)
- **Email:** [yeswantharasavalli@gmail.com](yeswantharasavalli@gmail.com)
- **Hugging Face:** [YeswanthArasavalli](https://huggingface.co/YeswanthArasavalli)
- **Project Issues:** [GitHub Issues](https://github.com/YeswanthArasavalli/Amazon-Food-Review-Sentiment-Analysis/issues)
