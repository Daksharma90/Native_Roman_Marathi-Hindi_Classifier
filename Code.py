from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer from Hugging Face
model_name = "GautamDaksh/Native_Marathi_Hindi_English_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def classify_sentence(sentence):
    words = sentence.split()
    predictions = []
    label_map_inv = {0: "E", 1: "H", 2: "M"}  # Reverse mapping for labels

    for word in words:
        inputs = tokenizer(word, return_tensors="pt", truncation=True, padding="max_length", max_length=32).to(device)
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
        predictions.append((word, label_map_inv[predicted_class]))

    return predictions

sentence = "रात्र असो का दिवस काही फरक पडत नाही"
print(classify_sentence(sentence))
