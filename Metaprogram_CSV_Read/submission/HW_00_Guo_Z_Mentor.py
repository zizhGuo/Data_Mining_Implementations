from datetime import datetime

def writeFile(trainedinfo):
    # create an empty string called "str" for reserving codes to be written into trained program
    str = ""

    # Writes in necessary libs and date to trained program
    str += f"import numpy as np"
    str += f"\nimport pandas as pd"
    str += f"\nfrom datetime import datetime"
    str += f"\n\nif __name__ == \"__main__\":"
    str += f"\n\tcreateDate = \"{datetime.now()}\""
    str += f"\n\tprint(f\"File Created date:    {{createDate}}\")       # time and date when this file created" 
    str += f"\n\tprint(f\"Current running date: {{datetime.now()}}\")   # current time and data"  

    # Writes in value of "name" key from info to trained program
    if "name" in trainedinfo.keys(): 
        str+= f"\n\tprint(\"First/Last name: { trainedinfo['name'] }\")               # My first name and last name"

    # Writes in CSV file name to trained program
    if "fileToRead" in trainedinfo.keys():
        str += f"\n\tdf = pd.read_csv(\"{trainedinfo['fileToRead']}\")        # read dataframe from the given CSV file"
        str += f"\n\tprint(\"rows:       \" + str(df.shape[1]))           # number of rows"
        str += f"\n\tprint(\"columns:    \" + str(df.shape[0] + 1))       # number of columns including headers"          

    # Writes in interesting personal info to trained program
    if "personalInfo" in trainedinfo.keys():
        str += f"\n\tprint(\"\"\"My hobbies: {trainedinfo['personalInfo']}\"\"\") "
    
    # for testing usage
    print(str)               
    # create a file with given name "filepath"
    assert "trainedProgramPath" in trainedinfo.keys()
    f = open(trainedinfo["trainedProgramPath"], "w") 
    # write string to the filepath 
    f.write(str)            
    f.close()
    

def trainProgram():
    # trained information
    trainedinfo = {
        "trainedProgramPath": "HW_00_ZG_Trained.py",
        "name": "Zizhun Guo",
        "fileToRead": "A_DATA_FILE.csv",
        "personalInfo": """ 
                        I do solo adventure usually accommodated in hostels.
                        I like discovering local neignborhood to experience cultural vibe. 
                        Versailles Palace is beautiful. Naples's Pizza is the best.
                        DC people are sharp on their wear choice, which is good.
                        I am from Chengdu, a place is known for spicy foods and girls.
                        We walk pandas on the street, and you can adopt it if you are rich.
                        I survive by cooking. Mostly Authentic Szchuan Cuisine for three years.  
                        I read books. Recenly travel essays and western view points on China
                        I watch movies. Recently topics about human relationships
                        One information above is not correct. Guess which one"""
    }
    # produce a trained program
    writeFile(trainedinfo)

if __name__ == "__main__":
    trainProgram()

    