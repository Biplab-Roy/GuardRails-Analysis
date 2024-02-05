import pandas as pd

dataframe = pd.read_csv("./questions.csv")
print(dataframe.content_policy_name.unique())
for policy_name in dataframe.content_policy_name.unique():
    df = dataframe[dataframe["content_policy_name"] == policy_name]
    df.to_csv(f'{policy_name}.csv', index = False)
    print("file ", policy_name, " : ", len(df))