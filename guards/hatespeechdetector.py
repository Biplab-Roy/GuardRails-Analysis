from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class HateSpeechDetector():
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/HateSpeechDetector")
        self.tokenizer = AutoTokenizer.from_pretrained("KoalaAI/HateSpeechDetector")
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        label = self.model.config.id2label[predicted_label]
        return {"status" : int(label != 'non-hate'), "explanation" : ''}
