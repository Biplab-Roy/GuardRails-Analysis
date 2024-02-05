from gemini import genaiClass
from llama_guard_7b import LLamaGuard
from beaver_dam_7b import beaver_dam_7b
from safety_flan_t5_base import SafetyFlanT5Base
from text_moderation import text_moderation
from one_for_all_toxicity_v3 import one_for_all_toxicity_v3
from moderation_23w48 import moderation
from distilroberta_base_rejection_v1 import distilroberta_base_rejection_v1
from offensiveSpeechDetector import OffensiveSpeechDetector
from hatespeechdetector import HateSpeechDetector
from flask import Flask
app = Flask(__name__)
bot = genaiClass()

@app.route('/')
def hello():
    return 'Hello, World!'

