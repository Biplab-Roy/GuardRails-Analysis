from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json

def truncate_illegal_json(json_str):
    # Try to load the JSON
    try:
        json_data = json.loads(json_str)
        # If loading succeeds, dump it again to ensure it's a valid JSON string
        truncated_json = json.dumps(json_data)
        return truncated_json
    except json.JSONDecodeError as e:
        # If JSON decoding fails, truncate the string before the error position
        truncated_json = json_str[:e.pos]
        return truncated_json

class moderation():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name_or_path = "UltroMi/moderation"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(self.device)
    
    def predict(self, text): 
        prompt = '{"text": "' + f'{text}"'
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        out = self.model.generate(input_ids, 
                            do_sample=True,
                            temperature=0.5,
                            top_k=5,
                            max_length=512
                            )
        # print(self.tokenizer.decode(out))
        generated_text = ' '.join(map(self.tokenizer.decode, out))
        generated_text = json.loads(truncate_illegal_json(generated_text))
        prediction = generated_text['flagged']
        return {"status" : int(prediction), "explanation" : json.dumps(generated_text,indent=4)}
