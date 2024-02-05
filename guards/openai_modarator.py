from openai import OpenAI

class openai_modarator():
    def __init__(self):
        self.client = OpenAI(api_key = "")

    def predict(self, text):
        response = self.client.moderations.create(input=text)
        output = response.results[0]["flagged"]
        return {"status" : int(output != True), "explanation" : response.results[0]}
    
# op = openai_modarator()
# print(op.predict("Here I am"))