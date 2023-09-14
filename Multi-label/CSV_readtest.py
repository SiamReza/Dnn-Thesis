import pandas as pd
col = []
for i in range(1, 91):
	col.append('Q{}'.format(i))
df = pd.read_csv('SCL90.csv', on_bad_lines='skip', usecols = col, sep=';')
df = df.dropna()

print(df.shape)
df2 = df.iloc[1:30]
print(df.shape)
print(df2.shape)
