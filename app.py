import torch
import zipfile
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import gradio as gr

# unzip model if needed
if "distilbert_amazon_food" not in os.listdir():
    with zipfile.ZipFile("distilbert_amazon_food.zip", "r") as f:
        f.extractall(".")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert_amazon_food")
model = DistilBertForSequenceClassification.from_pretrained("distilbert_amazon_food").to(device)

def predict(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    logits = model(**enc.to(device)).logits
    prob = torch.softmax(logits, dim=1)
    label = torch.argmax(prob).item()
    return ["Negative", "Positive"][label] + f" ({prob[0][label]:.2f})"

iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Amazon Food Review Sentiment Analysis")
iface.launch()
