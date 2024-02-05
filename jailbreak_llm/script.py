models = [
    'llama', 
    'beaver', 
    'flan_t5_base', 
    'text_moderator', 
    'toxicity', 
    'moderator_23w48',
    'distilroberta',
    'offensiveClassifier',
    'hateClassifier'
]
files = ['Illegal Activity.csv', 'Hate Speech.csv', 'Malware.csv', 'Physical Harm.csv',
 'Economic Harm.csv', 'Fraud.csv', 'Pornography.csv', 'Political Lobbying.csv',
 'Privacy Violence.csv', 'Legal Opinion.csv', 'Financial Advice.csv',
 'Health Consultation.csv', 'Gov Decision.csv']
with open("script.sh", "w") as f:
    for file in files:
        for model in models:
            f.write(
    f'python "../guards/driver.py" -g {model} -d "../jailbreak_llm/{file}" | tee outs/{model}_{file.replace(" ", "_").split(".")[0]}.out\n'
            )
