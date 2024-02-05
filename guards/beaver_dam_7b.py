import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class beaver_dam_7b():
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sm = torch.nn.Softmax(dim = 1)
        self.tokenizer = AutoTokenizer.from_pretrained("PKU-Alignment/beaver-dam-7b")
        self.model = AutoModelForSequenceClassification.from_pretrained("PKU-Alignment/beaver-dam-7b").to(self.device)

    def predict(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prob = self.sm(logits).max().item()
        predicted_label = torch.argmax(logits, dim=1).item()

        # Return status 0, if there is no problem
        if prob >= 0.70:
            return {"status" : 1, "explanation" : self.model.config.id2label[predicted_label]}
        else:
            return {"status" : 0, "explanation" : ''}