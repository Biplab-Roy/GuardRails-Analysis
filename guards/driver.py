import warnings
warnings.filterwarnings('ignore')


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
import pandas as pd
from tqdm import tqdm 
import time
import os
import argparse

model = None
bot = None
prompt = None

parser = argparse.ArgumentParser(
                    prog='Run a guardrail',
                    description='Run your guardrail',
                    epilog='Text at the bottom of help')

parser.add_argument('-b', '--bot', choices=['genai', 'none'], default='none')
parser.add_argument('-g', '--guard', choices=['llama', 
                                              'beaver', 
                                              'flan_t5_base', 
                                              'text_moderator', 
                                              'toxicity', 
                                              'moderator_23w48',
                                              'distilroberta',
                                              'offensiveClassifier',
                                              'hateClassifier'
                                              ], required=True)
g = parser.add_mutually_exclusive_group(required=True)
g.add_argument('-d', '--dataset')
g.add_argument('-m', '--message')

args = parser.parse_args()

bot_on = (args.bot == 'genai')
if args.guard == 'llama':
    model = LLamaGuard()
elif args.guard == 'beaver':
    model = beaver_dam_7b()
elif args.guard == 'flan_t5_base':
    model = SafetyFlanT5Base()
elif args.guard == 'text_moderator':
    model = text_moderation()
elif args.guard == 'toxicity':
    model = one_for_all_toxicity_v3()
elif args.guard == 'moderator_23w48':
    model = moderation()
elif args.guard == 'distilroberta':
    model = distilroberta_base_rejection_v1()
elif args.guard == 'offensiveClassifier':
    model = OffensiveSpeechDetector()
elif args.guard == 'hateClassifier':
    model = HateSpeechDetector()


def evaluate_csv(df, model, bot):
    rejects = 0
    for ind, row in tqdm(df.iterrows()):
        prompt = row["text"]
        if bot:
            response = bot.answer(row["text"])
            prompt = str({"user" : row['text'], "bot" : response})
        classification = model.predict(prompt)
        if(classification["status"] == 1):
            rejects += 1
    return (rejects, len(df))
    
def evaluate_msg(msg, model, bot):
    prompt = msg
    if bot:
        response = bot.answer(msg)
        prompt = str({"user" : msg, "bot" : response})
    classification = model.predict(prompt)
    return (classification["status"], 1)

if bot_on:
    bot = genaiClass()

if args.dataset != None:
    df = pd.read_csv(args.dataset)
    result = evaluate_csv(df, model, bot)
else:
    result = evaluate_msg(args.message, model, bot)

print(f"{result[0]} prompts rejected out of {result[1]}")

print(f"Total percentage of rejection: {result[0] / result[1]}")
print(f"Total percentage of acceptance: {1 - (result[0] / result[1])}")