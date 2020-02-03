import numpy as np
import pandas as pd
from datetime import datetime

if __name__ == "__main__":
	createDate = "2020-01-26 21:55:37.722122"
	print(f"File Created date:    {createDate}")       # time and date when this file created
	print(f"Current running date: {datetime.now()}")   # current time and data
	print("First/Last name: Zizhun Guo")               # My first name and last name
	df = pd.read_csv("A_DATA_FILE.csv")        # read dataframe from the given CSV file
	print("rows:       " + str(df.shape[1]))           # number of rows
	print("columns:    " + str(df.shape[0] + 1))       # number of columns including headers
	print("""My hobbies:  
                        I do solo adventure usually accommodated in hostels.
                        I like discovering local neignborhood to experience cultural vibe. 
                        Versailles Palace is beautiful. Naples's Pizza is the best.
                        DC people are sharp on their wear choice, which is good.
                        I am from Chengdu, a place is known for spicy foods and girls.
                        We walk pandas on the street, and you can adopt it if you are rich.
                        I survive by cooking. Mostly Authentic Szchuan Cuisine for three years.  
                        I read books. Recenly travel essays and western view points on China
                        I watch movies. Recently topics about human relationships
                        One information above is not correct. Guess which one""") 