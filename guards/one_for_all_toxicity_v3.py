import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class one_for_all_toxicity_v3():
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("FredZhang7/one-for-all-toxicity-v3")
        self.model = AutoModelForSequenceClassification.from_pretrained("FredZhang7/one-for-all-toxicity-v3").to(self.device)

    def predict(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding = True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1).item()
        # Return status 0, if there is no problem
        return {"status" : predicted_labels, "explanation" : ''} 