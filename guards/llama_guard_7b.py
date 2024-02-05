from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLamaGuard():
    def __init__(self):
        model_id = "meta-llama/LlamaGuard-7b"
        self.device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=self.device)

    def predict(self, text):
        chat = [
            {"role": "user", "content": text},
        ]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

# llamaGuard = LLamaGuard()
# res = llamaGuard.moderate([
#     {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
#     {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
# ])
# res = llamaGuard.moderate("Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate.")
# `safe`

# print(res)