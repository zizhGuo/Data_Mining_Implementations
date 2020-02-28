# -----------------------------------------------------------
# CSCI-720-Big-Data-Analytics
# Assignment: HW03
#
# Author: Zizhun Guo
# Email: zg2808@cs.rit.edu
# 
# RIT, Rochester, NY
# 
# Zizhun GUO @ All rights researved
# -----------------------------------------------------------

import numpy as np # fundamental scientific computing package
import pandas as pd # data analysis and manipulation package
import matplotlib.pyplot as plt # plotting package
from datetime import datetime # python std data package
import sys # for using command line argument


def cross_corelation(data_x, data_y):
    """Calculate the cross-corelation with two lists
    Paras:
        @data_x: a list of attribute data
        @data_y: a list of target data
    Return:
        void
    """
    mean_x = np.mean(data_x)            # mean x
    mean_y = np.mean(data_y)            # mean y
    std_x = np.std(data_x)              # standard deviation x
    std_y = np.std(data_y)              # standard deviation y
    size_x = np.size(data_x)
    size_y = np.size(data_y)

    assert size_x == size_y
    correlation = np.sum(               # N-dimensional Cross-Correlation
        (data_x - mean_x) 
        * (data_y - mean_y) 
        / std_x 
        / std_y) \
        / size_x
    return correlation

def attribute_selection(correlations_tuples):
    """Select the attribute with highest absolute of cross-correlation
    Paras:
        @correlations_tuples: a list of tuples contains cross-correlations
                            and matching attribute
    Return:
        best_correlation: a floatof highest correlation
        best_attribute: a string of attribute name
    """
    best_attribute = ''
    best_correlation = 0
    # print(correlations_tuples)

    # gets greatest correlation as best attirbute by comparing mod value
    for corelation_tuple_with_tags in correlations_tuples:
        current_correlation = corelation_tuple_with_tags[0]
        current_attribute = corelation_tuple_with_tags[1]
        if np.abs(np.abs(current_correlation) > np.abs(best_correlation)):
            best_correlation = current_correlation
            best_attribute = current_attribute
    print(correlations_tuples)
    print("The highest cross-correlation:  " + str(best_correlation))
    print("The best feature/attribute:     " + best_attribute)
    return best_correlation, best_attribute

def data_preprocessing(dataframe_data, series_target):
    """Pre-processing the data and convert it into correlation tuples
    Paras:
        @dataframe_data: a dataframe of data read from CSV file
        @series_target: a list of int representing the target variable
    Return:
        correlations_tuples: a list of tuples contains cross-correlations
                            and matching attribute
    """
    # initialization   
    correlations_tuples = []

    # sets colunm names' size
    column_names_size = np.size(dataframe_data.columns)

    # label slice to keep only attributes but target varaible
    column_names = dataframe_data.columns[1:column_names_size - 1]

    # calculate cross-correlation for each attribute with target var
    for column_name in column_names:
       correlations_tuples.append(
        (cross_corelation
            (dataframe_data[column_name], series_target), 
            column_name))
    
    # initialize row/column count
    row_count = dataframe_data.shape[0]           
    column_count = dataframe_data.shape[1]

    # calculate the row/column count with at least one value assigned
    for row_index in range(0, dataframe_data.shape[0]):
        count = 0
        for column_index in range(0, dataframe_data.shape[1]):
            if pd.isna(dataframe_data.iloc[row_index, column_index]):
                count += 1
        if count == dataframe_data.shape[1]: # if all elements are NaN
            row_count -= 1

    for column in dataframe_data.columns:
        if(dataframe_data[column].dropna().empty):
            column_count -= 1

    return correlations_tuples, row_count, column_count

def ditribution_plot_with_jitter(df_filtered, best_attribute, target_variable, size):
    """ Draw distribution figure of (attribute, target) pairs' dot
    Paras:
        @df_filtered: a DATAPFRAME maintaing two columns seires: selected 
                        bsest attribute and target attribute
        @best_attribute: a string that contains the name of best attribute
        @target_variable: a string that contains the name of targetvariable
    Return:
        void
    """
    scatter_fraction_rate = 0.3                     # scatter fraction rate fator
    scatter_scale = 10                              # scale factor
    
    # x = df_filtered[best_attribute] * scatter_scale
    # y = df_filtered[target_variable]* scatter_scale

    # scale x-data and adds jitter
    x = df_filtered[best_attribute] \
                * (1 - scatter_fraction_rate) \
                * scatter_scale \
            + np.random.ranf(size) \
                * scatter_fraction_rate \
                * scatter_scale
    
    # scale y-data and adds jitter
    y = df_filtered[target_variable] \
                * (1 - scatter_fraction_rate) \
                * scatter_scale \
            + np.random.ranf(size) \
                * scatter_fraction_rate \
                * scatter_scale

    plt.scatter(x, y, alpha=0.5)                # plot the dots distribution
    # plt.title('Scatter situation ')      # sets title
    plt.title('Scatter situation +jitter')      # sets title
    plt.xlabel(best_attribute)                  # sets x labels
    plt.ylabel(target_variable)                 # sets y labels
    plt.show()                                  # show figure

def write_file(best_attribute, is_positive_correlated):
    """ writes trained program based on selected best attribute
    Paras:
        @best_attribute: a string represents the name of best attribute
        @is_positive_correlated: a boolean determines the classification rule
    Return:
        void  
    """
    # initialize a string to contains codes for dumping in the trained program
    lines = ""

    # add packages
    lines += f"import numpy as np"
    lines += f"\nimport pandas as pd"
    lines += f"\nfrom datetime import datetime"
    lines += f"\nimport sys"

    # add main functions
    # read validation csv file into data and print the created time & current running time
    lines += f"\n\ndef main():"
    lines += f"\n\tdf_food_poisoning = pd.read_csv(\'../\' + sys.argv[1])"
    # lines += f"\n\tdf_food_poisoning = pd.read_csv(\"Food_Poisoning_Data_v3.csv\")"
    lines += f"\n\tbest_attribute = \'{best_attribute}\'"
    lines += f"\n\tcreateDate = \"{datetime.now()}\""
    lines += f"\n\tprint(f\"File Created date:    {{createDate}}\")       # time and date when this file created" 
    lines += f"\n\tprint(f\"Current running date: {{datetime.now()}}\")   # current time and data"  
        
    # 
    lines += f"\n\tdata = df_food_poisoning[best_attribute]"

    if is_positive_correlated:   
        #               
        lines += f"\n\tfor val in data:"
        lines += f"\n\t\tif val >  0:"
        lines += f"\n\t\t\tprint('1')"
        lines += f"\n\t\telse:"
        lines +=  f"\n\t\t\tprint('0')"
    else:            
        lines += f"\n\tfor val in data:"
        lines += f"\n\t\tif val ==  0:"
        lines += f"\n\t\t\tprint('1')"
        lines += f"\n\t\telse:"
        lines +=  f"\n\t\t\tprint('0')"       

    lines += f"\n\nif __name__ == \"__main__\":"
    lines += f"\n\tmain()"

    # for testing usage
    # print(lines)               
    
    # create a file with given name "filepath"
    f = open('HW_03_Guo_Zizhun_Trained.py', "w") 
    # write string to the filepath 
    f.write(lines)            
    f.close()    


def main():
    """ Main program that does these things:
        - select best feature/attribute from correlation value tuples
        - print out frequency table
        - draw distribution plot with jitter
        - write trained file
    """
    # df_food_poisoning = pd.read_csv('Food_Poisoning_Data_v3.csv')
    
    # sets the CSV file path
    df_food_poisoning = pd.read_csv('../' + sys.argv[1]) # Requirment

    # sets the target variable name as 'Sickness'
    target_variable = 'Sickness'

    # initialize the row/columns size for data preprocessing
    rows_size = np.inf
    columns_size = np.inf

    # intialize a Series representing the target series 
    # in dataframe where columns equals to target name
    data_target = df_food_poisoning[target_variable]

    # data preprocessing and gets all correlations 
    # stored in a list of tuples for feature selection
    correlations_tuples, rows_size, columns_size \
        = data_preprocessing(df_food_poisoning, data_target)
    
    # select the best feature/attribute
    best_correlation, best_attribute = attribute_selection(correlations_tuples)

    # sets flag variable based on value of the best correlation
    is_positive_correlated = True if best_correlation > 0 else False

    # Keep data only from columns of best_attribute and target_variable
    df_filtered = df_food_poisoning.loc[:, [best_attribute, target_variable]]
    
    # groupby the dataframe to get frequency table
    frequency_table = df_filtered.groupby(
        [best_attribute, target_variable], as_index=False) \
            .size()
    
    # print the frequence table
    print(frequency_table)
    
    # draw the distribution figure to observe the cross-correlation
    ditribution_plot_with_jitter(df_filtered, best_attribute, 
                                target_variable, 
                                rows_size)

    # write the trained the file 
    write_file(best_attribute, is_positive_correlated)   
    
    # ignore test functions below plz
    # XOR = [df_food_poisoning[best_attribute] ^ data_Sickness] # exclusive OR: XOR
    # print(XOR)     
    # print(grouped)
    # print(grouped[0][0])
    # print(df_filtered)

if __name__ == "__main__":
    main()