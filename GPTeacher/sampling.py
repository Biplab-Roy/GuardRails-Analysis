import pandas as pd

dataframe = pd.read_csv("./dataset.csv")
sample = dataframe.sample(frac = 0.05)

for file in sample.filename.unique():
    df = sample[sample["filename"] == file]
    df.to_csv(f'{file}.csv', index = False)
    print("file ", file, " : ", len(df))