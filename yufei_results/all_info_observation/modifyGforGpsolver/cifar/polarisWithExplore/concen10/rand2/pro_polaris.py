import pandas as pd

df = pd.read_csv('polaris.csv')

df.loc[0] = ['Amy', 89, 93] 