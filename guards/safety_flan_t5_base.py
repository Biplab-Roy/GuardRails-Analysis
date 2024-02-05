from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
import torch

class SafetyFlanT5Base:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = AutoConfig.from_pretrained("Salesforce/safety-flan-t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/safety-flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/safety-flan-t5-base", config=self.config).to(self.device)

    def predict(self, input_text, input_context = ''):
        prefix = "Is the <Text> field safe or unsafe?"
        input_ids = self.tokenizer(prefix + " <Text> " + input_text + " <Context> " + input_context, return_tensors="pt").input_ids.to(self.device)

        outputs = self.model.generate(input_ids, max_length = 1024)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"status" : int(result.split(' ')[0] != 'safe'), "explanation" : ''}
