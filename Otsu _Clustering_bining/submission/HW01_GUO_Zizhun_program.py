# -----------------------------------------------------------
# CSCI-720-Big-Data-Analytics-HW01
#
# 2020 RIT, Rochester, NY
#
# Author: Zizhun Guo
# 
# Email zg2808@cs.rit.edu
#
# All rights researved
# -----------------------------------------------------------

import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np

def mystery_data():
    """Explore mystery data"""

    # Read raw data as pandas' DataFrame
    mystery_data_raw = pd.read_csv('Mystery_Data_2195.csv') 
    
    # print the tail of data set
    print(mystery_data_raw.tail())

    # Get column data as pandas Series
    mystery_data = mystery_data_raw[
                        mystery_data_raw.columns[0]]

    # using numpy package to calculate the average and standard deviation
    print("(before) Average =            " + str(np.mean(mystery_data).astype(float)))
    print("(before) Standard Deviation = " + str(np.std(mystery_data).astype(float))) 
   
    # Remove the last value from the data
    mystery_data_removed = mystery_data[:len(mystery_data)]

    # using numpy package to calculate the average and standard deviation
    print("(after) Average =             " + str(np.mean(mystery_data_removed)
                                                .astype(float)))
    print("(after) Standard Deviation =  " + str(np.std(mystery_data_removed)
                                                .astype(float))) 

def otsu(data, is_regularization, offset, alpha):
    """ otsu clustering
    
    Keyword arguments:
    data -- the array contains quantized bined raw data
    is_regularization -- the boolean variable if do regularization
    offset -- the float of bin size
    alpha -- alphas for comparing splitting point
    
    Return: 
    best_cost -- the int of best splitting point
    """
    start = np.min(data) + offset           # start point
    end = np.max(data)                      # end point

    # for graphing snowfolk's data based on quantized age/height
    mix_variance = []
    quantized_data = np.arange(start, end + offset, offset)

    best_cost = np.inf                      # best cost
    best_threshold = start                  # best splitting point

    while start <= end:
        wt_left = np.sum(data[data < start])/np.sum(data)     # fraction left
        wt_right = np.sum(data[data >= start])/np.sum(data)   # fration right
        wt_var_left = np.var(data[data < start])              # variance left
        wt_var_right = np.var(data[data >= start])            # variance right

        mixed_variance = wt_left * wt_var_left \
                        + wt_right * wt_var_right             # mixed variance
        
        if is_regularization:                                 
            mix_cost = mixed_variance \
                    + abs(np.sum(data[data < start]) \
                            - np.sum(data[data >= start])) \
                        / 100 * alpha         # assign mixed cost to otsu' cost
        else:
            mix_cost = mixed_variance         # assign mixed variance to otsu' cost
            mix_variance.append(mix_cost)
               
        if (mix_cost <= best_cost):           # assign the minimal mixed cost as otsu' cost
            best_cost = mix_cost
            best_threshold = start

        start += offset                       # splitting point moves on
    print(best_cost)
    
    # test all alphas for comparing among different best splitting point
    if is_regularization is False:            
        assert mix_variance
        plt.plot(quantized_data, mix_variance)                            # plot lines by connecting points
        plt.bar(quantized_data, mix_variance)                             # plot bars of histogram
        plt.axvline(best_threshold, color = 'r')                          # indicate the segment value of point
        plt.title(f'Best threshold: {best_threshold}),[{best_threshold}') # add the title indicating the best point
        plt.ylabel('Mixed Variance')                                      # add label of y-axis
        plt.xlabel("Quantized Data")                                      # add label of x-axis
        plt.show()                                                        # show the figure
    
    return best_threshold       # return the best splitting point


def program(data, is_regularization, offset):
    """ Program for using otsu method and graphing
    
    Keyword arguments:
    data -- the array contains quantized bined raw data
    is_regularization -- the boolean variable if do regularization
    offset -- the float of bin size
    """
    alphas = [100, 1, 1/5, 1/10, 1/20, 1/25, 1/50, 1/100, 1/1000]                 # list of all alphas
    
    # for graphing best splitting point based on different alphas
    best_thresholds = []

    if is_regularization:
        for alpha in alphas:                                                      # traverse all alphas
            best_thresholds.append(otsu(data, is_regularization, offset, alpha))  # call otsu
        
        # collect all best splitting point under different value of alphas
        df = pd.DataFrame(data = best_thresholds, 
                            index = [str(alpha) for alpha in alphas], 
                            columns = {'Best_threshold'})

        # use Matplotlib package to draw the graph               
        plt.plot(df.index, df["Best_threshold"])                          # draw the graph
        plt.ylabel('Best splitting point')                                      # add label of y-axis
        plt.xlabel("Alphas")                                      # add label of x-axis
        plt.show()                                                        # show the graph
    else:
        otsu(data, is_regularization, offset, None)                       # call otsu       

def main():
    """ Read snowfolks data to clustering it into two groups using otsu method"""
    
    # -----------------------------------------------------------------------------
    #
    # Section 1: Mystery Data exploration
    #
    # -----------------------------------------------------------------------------
    mystery_data() # call to explore mystery data

    # -----------------------------------------------------------------------------
    #
    # Section 2: Mystery Data exploration
    #
    # -----------------------------------------------------------------------------

    # Using pandas package to read Parse CSV file into Pandas DataFrame
    df_snowfolks_data_raw = pd.read_csv('Abominable_Data_For_Clustering__v44.csv')
    
    # Quantizing data using Bining method
    # Bined quatized data of snowfolks based on "Age" with step size of "2"
    data_ages= np.floor(df_snowfolks_data_raw['Age'] / 2) * 2

    # Bined quatized data of snowfolks based on "Height" with step size of "5"
    data_heights= np.floor(df_snowfolks_data_raw['Ht'] / 5) * 5  

    # testcase 1: data set on Age, no regularization, bin size 2
    program(data = data_ages, is_regularization = False, offset = 2)  

    # testcase 2: data set on Age, regularization, bin size 2
    program(data_ages, True, 2)

    # testcase 3: data set on Heights, no regularization, bin size 5
    program(data_heights, False, 5)

    # testcase 4: data set on Heights, regularization, bin size 5
    program(data_heights, True, 5)

if __name__ == "__main__":
    main()
