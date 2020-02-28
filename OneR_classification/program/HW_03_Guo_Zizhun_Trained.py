import numpy as np
import pandas as pd
from datetime import datetime
import sys

def main():
	df_food_poisoning = pd.read_csv('../' + sys.argv[1])
	best_attribute = 'PeanutButter'
	createDate = "2020-02-15 16:40:17.267135"
	print(f"File Created date:    {createDate}")       # time and date when this file created
	print(f"Current running date: {datetime.now()}")   # current time and data
	data = df_food_poisoning[best_attribute]
	for val in data:
		if val >  0:
			print('1')
		else:
			print('0')

if __name__ == "__main__":
	main()