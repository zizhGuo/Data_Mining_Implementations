import numpy as np
import pandas as pd
from datetime import datetime
from datetime import datetime
import sys

def main():
	df_snowfolks_data_raw = pd.read_csv('../' + sys.argv[1])
	createDate = "2020-02-09 11:39:26.124143"
	print(f"File Created date:    {createDate}")       # time and date when this file created
	print(f"Current running date: {datetime.now()}")   # current time and data
	attribute = "Height"
	threshold = [140.0, 140.0]
	data = df_snowfolks_data_raw[attribute]
	if attribute == 'Age':
		for val in data:
			if val <= threshold[0]:
				print('- 1')
			else:
				print('+ 1')
	if attribute == 'Height':
		for val in data:
			if val <= threshold[np.size(threshold) - 1]:
				print('+ 1')
			else:
				print('- 1')

if __name__ == "__main__":
	main()