import google.generativeai as genai

class genaiClass():
    def __init__(self) -> None:
        genai.configure(api_key=open("./GOOGLE_API_KEY", "r").read())
        self.model = genai.GenerativeModel('gemini-pro')

    def answer(self, question):
        try:
            response = self.model.generate_content(question)
            return response.text
        except:
            return "Blocked"