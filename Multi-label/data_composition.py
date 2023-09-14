import os

import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict

def main():
	disease = ['disease']
	phase = ['phase']
	df = pd.read_csv('SCL90Cleaned.csv', header = 0, on_bad_lines='skip', usecols = disease + phase)
	df = df.dropna()
	conditions = np.unique(df[disease].to_numpy())
	latex_table = defaultdict(list)
	latex_table['Condition'] = ['Sick', 'Healthy']
	for condition in conditions:
		sub_disease = df[df[disease[0]] == condition]
		latex_table[condition].append(len(sub_disease[sub_disease[phase[0]] != 'Utskriving']))
		latex_table[condition].append(len(sub_disease[sub_disease[phase[0]] == 'Utskriving']))
		
	latex_table['Total'].append(len(df[df[phase[0]] != 'Utskriving']))
	latex_table['Total'].append(len(df[df[phase[0]] == 'Utskriving']))
	latex_table = pd.DataFrame(latex_table)
	latex_table = latex_table.to_latex(index = False, caption = 'Overall data composition', position = 'H')
	with open("./Result/data_composition.tex", "w") as f:
		f.write(latex_table)
		
if __name__ == '__main__':
	main()
