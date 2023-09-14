import pandas as pd
import numpy as np
label = ['treatment type name', 'assessment instance first time submitted date', 'treatment type id']
phase = ['assessment instance context label']
respondent_id = ['respondent id']
feature = []
for i in range(1, 91):
	feature.append('Q{}'.format(i))
df = pd.read_csv('SCL90.csv', on_bad_lines='skip', usecols = label + phase + feature + respondent_id, sep=';')

df = df.rename({'treatment type name': 'disease', 'assessment instance context label': 'phase', 'respondent id': 'id', 'assessment instance first time submitted date': 'date', 'treatment type id':'code'}, axis = 'columns')


df = df.replace('Behandlingsstart', 'Innkomst', regex=True)
df = df.replace('Underveis', 'Utredning', regex=True) 
df = df.replace('Behandlingsstart', 'Utskriving', regex=True) 
df = df[df['phase'].isin(['Utredning', 'Innkomst', 'Utskriving'])]
diseases = df['disease'].unique()
diseases = list(diseases)
for d in diseases:
	if 'Poliklinikk' in d:
		diseases.remove(d)

df = df[df['disease'].isin(diseases)]
df['date'] = pd.to_datetime(df['date']).dt.date
df = df.replace(' ', '_', regex=True)	
df = df.dropna()
df.to_csv('SCL90Cleaned.csv')
