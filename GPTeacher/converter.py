import json
import pandas as pd

files = ['Codegen/codegen-instruct.json', 
         'Instruct/gpt4-instruct-similarity-0.9-dataset.json',
         'Roleplay Supplemental/roleplay-instruct-v2.1.json',
         'Toolformer/toolformer-similarity-0.9-dataset.json'
         ]

dataset = []
for file in files:
    content = json.load(open(file))
    filename = file.split("/")[0]
    for row in content:
        try:
            dataset.append([filename, row['instruction'] + row['input'], row['response']])
        except:
            try:
                dataset.append([filename, row['instruction'] + row['input'], row['output']])
            except:
                print(row.keys())
                print(filename)

dataframe = pd.DataFrame(dataset, columns = ['filename', 'text', 'response'])
dataframe.to_csv('dataset.csv', index = False)