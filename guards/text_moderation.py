import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class text_moderation():
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")
        self.model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation").to(self.device)
        self.categories = {
            'S': {'Category': 'sexual', 'Definition': 'Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).'},
            'H': {'Category': 'hate', 'Definition': 'Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.'},
            'V': {'Category': 'violence', 'Definition': 'Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.'},
            'HR': {'Category': 'harassment', 'Definition': 'Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.'},
            'SH': {'Category': 'self-harm', 'Definition': 'Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.'},
            'S3': {'Category': 'sexual/minors', 'Definition': 'Sexual content that includes an individual who is under 18 years old.'},
            'H2': {'Category': 'hate/threatening', 'Definition': 'Hateful content that also includes violence or serious harm towards the targeted group.'},
            'V2': {'Category': 'violence/graphic', 'Definition': 'Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.'},
            'OK': {'Category': 'OK', 'Definition': 'Not offensive'}
        }


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
            predicted_label = torch.argmax(logits, dim=1).item()

        label = self.model.config.id2label[predicted_label]

        # Return status 0, if there is no problem
        return {"status" : int(label != "OK"), "explanation" : self.categories[label]}
        