import os
import pandas as pd
import json
from collections import defaultdict

def create_table(dir = '.'):
	latex_tables = {}

	latex_tables= defaultdict(list)
	stat_files = os.listdir(dir)
	stat_files.sort()
	for stat_file in stat_files:
		if 'stats_' in stat_file: 
			method = stat_file.replace('stats_', '')
			method = method.replace('.json', '')
			latex_tables['method'].append(method)
			with open(os.path.join(dir, '{}'.format(stat_file))) as f:
				data = json.load(f)
				for key, value in data.items():
					if key != 'F0.5':
						latex_tables[key].append('{} '.format(round(value['mean'], 2)) + u'\u00B1' + '{}'.format(round(value['SD'], 2)))
							
	latex_table = pd.DataFrame(latex_tables)
	table = latex_table.to_latex(index = False)
	with open(os.path.join(dir, "Single label.tex"), "w") as f:
		f.write(table)

if __name__ == '__main__':
	create_table('.')
