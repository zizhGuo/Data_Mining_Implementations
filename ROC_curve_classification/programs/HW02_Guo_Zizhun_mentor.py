# -----------------------------------------------------------
# CSCI-720-Big-Data-Analytics
# Assignment: HW02
#
# Author: Zizhun Guo
# Email: zg2808@cs.rit.edu
# 
# RIT, Rochester, NY
# 
# ZG@All rights researved
# -----------------------------------------------------------

import numpy as np # fundamental scientific computing package
import pandas as pd # data analysis and manipulation package
import matplotlib.pyplot as plt # plotting package
from datetime import datetime # python std data package
import sys # for using command line argument

def hist(data, bin_size, type):
    """ Draw histogram on dataset
    Paras:      
        @data: a list signed dataset
        @bin_size: an int size of each bin
        @type: a string name of dataset
    Return:
        void
    """
    data_pos = data[data > 0]                               # dataset of Class: Bhutan
    data_neg = - data[data < 0]                             # dataset of Class: Assam
    num_bins = (np.max(data_pos)                            # # of bins
                + bin_size 
                - np.min(data_pos)) \
                / bin_size
    plt.hist([data_neg, data_pos],                          # plot histogram
            bins= int(num_bins),                            # # of bins
            density=False,
            stacked = True)                                 # stack two classes
    plt.title('Data for Silver-Backed Abominable Snowfolks') # sets title
    plt.xlabel(type + ' Feature based Distribution')         # sets x label
    plt.legend(['Assam', 'Bhutan'])                          # sets legend
    plt.show()                                               # render the graph

def ROC_curve(TPR_FPR_tuples, roc_points_with_lowest_cost):
    """ Draw Receiver Operating Characteristics(ROC) Curve
    Paras:      
        @TPR_FPR_tuples: a list of tuples containing pairs of TPR-FPR rates
        @roc_points_with_lowest_cost: a list of list which contains tuples of
                                    the coordinates of points of lowest costs
    Return:
        void
    """
    # Traverse all tuple to scatter and connects on the graph
    for TPR_FPR_tuple in TPR_FPR_tuples:
        plt.scatter(*zip(*TPR_FPR_tuple), s = 20)
        plt.plot(*zip(*TPR_FPR_tuple))
    
    # Mark the rate point with its matching threshold annotated
    # Data structure [[(x1, y1), threshold1],...,[(xn, yn), thresholdn]]
    for roc_point_with_lowest_cost in roc_points_with_lowest_cost:
        for roc_point_with_lowest_cost_duplicate in roc_point_with_lowest_cost:
            plt.plot(roc_point_with_lowest_cost_duplicate[0], 
                    roc_point_with_lowest_cost_duplicate[1], 
                    'ro')
            plt.annotate(roc_point_with_lowest_cost_duplicate[2], 
                        (roc_point_with_lowest_cost_duplicate[0], 
                        roc_point_with_lowest_cost_duplicate[1]))

    plt.xlim(0, 1)                                      # limits range of x-axis
    plt.ylim(0, 1)                                      # limits range of y-axis
    plt.gca().set_aspect('equal', adjustable='box')     # Axes squared
    plt.xticks(np.arange(0, 1.0, 0.1))                  # sets x ticks step size as 0.1
    plt.yticks(np.arange(0, 1.0, 0.1))                  # sets y ticks step size as 0.1
    plt.grid(True, linewidth = 0.5)                     # plot gird
    plt.title('ROC curve')                              # sets title
    plt.xlabel("False Positive Rate / FPR ->")          # sets x labels
    plt.ylabel('<- True Positive Rate / TPR')           # sets y labels
    plt.legend(['Age', 'Height', 'best thresholds',])   # sets legends
    plt.show()                                          # render the graph

def mixed_costs_threshold(thresholds, mixed_costs, best_thresholds, step, type):
    """ Draw mixed costs based on each thresholds with highlighted best thresholds
    Paras:      
        @thresholds: a list of all possible thresholds
        @mixed_costs: a list of all possible mixed costs
        @best_thresholds: a list of best thresholds (one or multiple)
        @step: an int of bin size
        @type: a string of current attribute name
    Return:
        void
    """
    plt.plot(thresholds, mixed_costs)                           # plot the line graph
    for best_threshold in best_thresholds:                      # highlight the best threshold
        plt.axvline(best_threshold, color = 'r')
    plt.xticks(np.arange                                        # sets the x ticks size as a step
                    (np.min(thresholds),                         
                    np.max(thresholds) + step, 
                    step))         
    for label_to_hide in plt.axes().xaxis.get_ticklabels()[::2]:        # hide x labels in every 2 steps
        label_to_hide.set_visible(False)
    plt.grid(True, linewidth = 0.5)                             # plot grid
    plt.title('Mixed Cost based on Attribute Thresholds')     # sets title
    plt.xlabel(type + ' Thresholds')                             # sets x label
    plt.ylabel('Mixed Costs')                 # sets y label
    plt.show()                                                  # render the graph

def binary_classifier_1d(data, step, classfication_rule_A_to_B):
    """ One Dimensional Binary Classifier
    Paras:      
        @data: a list of signed dataset
        @step: an int of bin size
        @classfication_rule_A_to_B: domain knowledge to regulate direction
                                    of how to classify classes:
                                    True: left is Assam, right is Bhutan
                                    False: left is Bhutan, right is Assam
    Return:
        thresholds: a list of all possible thresholds
        mixed_costs: a list of all possible mixed costs
        best_thresholds: a list of best thresholds (one or multiple)
        TPR_FPR_tuples: a list of tuples containing pairs of TPR-FPR rates
        roc_point_with_lowest_cost: a list of tuples of the coordinates of 
                                    points of lowest costs with matching
                                    threshold
    """
    start = np.floor(np.min(np.abs(data)))                  # sets start threshold
    end = np.ceil(np.max(np.abs(data)))                     # sets end threshold

    TPR_FPR_tuples = []                                     # initialization
    mixed_costs = []                                        # ...
    thresholds = []                                         # ...

    best_thresholds = []                                    # ...
    lowest_cost = np.inf                                      # ...
    roc_point_with_lowest_cost = []                         # ...

    # while loop traversing all thresholds & classify datasets into two classes
    # based on the current threshold
    while start <= end:                                    
        Class_A = data[np.abs(data) < start] \
            if classfication_rule_A_to_B \
            else data[np.abs(data) >= start]
        Class_B = data[np.abs(data) >= start] \
            if classfication_rule_A_to_B \
            else data[np.abs(data) < start]

        FN = np.size(Class_A[Class_A > 0])                  # calculate the False Negative
        FP = np.size(Class_B[Class_B < 0])                  # calculate the False Positives
        mixed_cost = FN + FP                                # calculate the mixed cost by adding up
        mixed_costs.append(mixed_cost)                      # store the current mixed cost
        thresholds.append(start)                            # store the current threshold
        # print(mixed_cost)
        TP = np.size(Class_B[Class_B > 0])                      # calculate the True Positive
        TPR_FPR_tuples.append((FP / np.size(data[data < 0]),    # calculate the ROC & store as tuple
                                TP / np.size(data[data > 0])))
        # Minimize the mistakes
        if mixed_cost < lowest_cost:
            lowest_cost = mixed_cost                              # update the minimal cost
            best_thresholds = [start]                           # update the best threshold
            roc_point_with_lowest_cost = [                      # update the roc point with lowest cost
                    (FP / np.size(data[data < 0]), 
                    TP / np.size(data[data > 0]), 
                    start)]

        if mixed_cost == lowest_cost:                             # for repeating minimal cost
            best_thresholds.append(start)                       #   store the new threshold
            roc_point_with_lowest_cost.append(                  #   update the roc point into list
                (FP / np.size(data[data < 0]), 
                TP / np.size(data[data > 0]), 
                start))
      
        start += step                                           # move to the next treshold
    # roc_point_with_lowest_cost.append((0.4583333333333333, 0.8142857142857143, 'test'))
    print("Best cost = " + str(lowest_cost) 
            + "  Best threshold = " 
            + str(best_thresholds))
    return thresholds, lowest_cost, mixed_costs, best_thresholds, \
            TPR_FPR_tuples, roc_point_with_lowest_cost

def write_file(best_thresholds_ages, best_thresholds_heights, 
                lowest_cost_ages, lowest_cost_heights):
    """ Writes a trained 
    Paras:      
        @best_thresholds_ages: a list of best thresholds of Age dataset
        @best_thresholds_heights: a list of best thresholds of Height dataset
        @lowest_cost_ages: an int of the lowest cost of Age dataset
        @lowest_cost_heights: an int of the lowest cost of Height dataset
    Return:
        void
    """
    # select attribute by comparing the lowest cost to decide the classify rule
    attribute = 'Age' \
        if lowest_cost_ages < lowest_cost_heights \
        else 'Height'
    threshold = best_thresholds_ages \
        if lowest_cost_ages < lowest_cost_heights \
        else best_thresholds_heights
    

    # initialize a string to contains codes for dumping in the trained program
    lines = ""

    # add packages
    lines += f"import numpy as np"
    lines += f"\nimport pandas as pd"
    lines += f"\nfrom datetime import datetime"
    lines += f"\nfrom datetime import datetime"
    lines += f"\nimport sys"

    # add main functions
    # read validation csv file into data and print the created time & current running time
    lines += f"\n\ndef main():"
    lines += f"\n\tdf_snowfolks_data_raw = pd.read_csv(\'../\' + sys.argv[1])"
    lines += f"\n\tcreateDate = \"{datetime.now()}\""
    lines += f"\n\tprint(f\"File Created date:    {{createDate}}\")       # time and date when this file created" 
    lines += f"\n\tprint(f\"Current running date: {{datetime.now()}}\")   # current time and data"  
    
    # assign the selected attribute & threshold to the string as the classifier
    lines += f"\n\tattribute = \"{attribute}\""
    lines += f"\n\tthreshold = {threshold}"
    
    # select the attribute based on given attribute
    lines += f"\n\tdata = df_snowfolks_data_raw[attribute]"

    # classification based on attribute and domain knowledge
    lines += f"\n\tif attribute == 'Age':"               
    lines += f"\n\t\tfor val in data:"
    lines += f"\n\t\t\tif val <= threshold[0]:"
    lines += f"\n\t\t\t\tprint('- 1')"
    lines += f"\n\t\t\telse:"
    lines +=  f"\n\t\t\t\tprint('+ 1')"
    
    lines += f"\n\tif attribute == 'Height':"
    lines += f"\n\t\tfor val in data:"
    lines += f"\n\t\t\tif val <= threshold[np.size(threshold) - 1]:"
    lines += f"\n\t\t\t\tprint('+ 1')"
    lines += f"\n\t\t\telse:"
    lines += f"\n\t\t\t\tprint('- 1')"

    lines += f"\n\nif __name__ == \"__main__\":"
    lines += f"\n\tmain()"

    # for testing usage
    print(lines)               
    
    # create a file with given name "filepath"
    f = open('HW02_GUO_Zizhun_trained.py', "w") 
    # write string to the filepath 
    f.write(lines)            
    f.close()

def main():
    """ Main function running the program
    """
    # read & convert the csv file into data DataFrame
    # df_snowfolks_data_raw = pd.read_csv('../Abominable_Data_For_1D_Classification__v93_HW3_720_final.csv')
    df_snowfolks_data_raw = pd.read_csv('../' + sys.argv[1])
    
    # Initialization for the public ROC tuples and ROC coordinates (with lowest cost) for ROC curve graph
    TPR_FPR_tuples = []
    roc_points_with_lowest_cost = []


    ####################################### 1. Age ###########################################
    
    # Initialization from for all necessary data strutures for training and plotting
    bin_size_ages = 2
    data_ages= []
    thresholds_ages= []
    lowest_cost_ages = np.inf
    mixed_costs_ages = []
    best_thresholds_ages = []
    TPR_FPR_tuples_age = []
    roc_point_with_lowest_cost_ages = []
    
    # Quantize the dataset using Bining Method
    data_ages = np.floor(df_snowfolks_data_raw['Age'] /bin_size_ages) \
                    * bin_size_ages  \
                    * df_snowfolks_data_raw['Class']

    # Threshold Setting for Age Data
    thresholds_ages, lowest_cost_ages, mixed_costs_ages, \
    best_thresholds_ages, TPR_FPR_tuples_age, \
    roc_point_with_lowest_cost_ages \
            = binary_classifier_1d(data_ages, bin_size_ages, True)
    
    # Store ROC tuple
    TPR_FPR_tuples.append(TPR_FPR_tuples_age)

    # Store the coordinate of the ROC point with matching thresholds
    roc_points_with_lowest_cost.append(roc_point_with_lowest_cost_ages)
    

    ######################################## 2. Heights ########################################
    
    # Initialization from for all necessary data strutures for training and plotting
    bin_size_heights = 5   
    data_heights = []
    thresholds_heights= []
    lowest_cost_heights = np.inf
    mixed_costs_heights = []
    best_thresholds_heights = np.inf
    TPR_FPR_tuples_heights = []
    roc_point_with_lowest_cost_heights = []
    
    # Quantize the dataset using Bining Method
    data_heights= np.floor(df_snowfolks_data_raw['Height'] /bin_size_heights) \
                        * bin_size_heights \
                        * df_snowfolks_data_raw['Class']
    
    # Threshold Setting for Age Data
    thresholds_heights, lowest_cost_heights, mixed_costs_heights, \
    best_thresholds_heights, TPR_FPR_tuples_heights, \
    roc_point_with_lowest_cost_heights \
        = binary_classifier_1d(data_heights, bin_size_heights, False)
    
    # Store ROC tuple
    TPR_FPR_tuples.append(TPR_FPR_tuples_heights)

    # Store the coordinate of the ROC point
    roc_points_with_lowest_cost.append(roc_point_with_lowest_cost_heights)
    

    ##################################### 3. Writing File ####################################

    write_file(best_thresholds_ages, best_thresholds_heights, 
                lowest_cost_ages, lowest_cost_heights)


    ###################################### 4. Rendering ######################################
    
    # call for histgram redering on "Age" data
    hist(data_ages, bin_size_ages, 'Age')                      

    # call for mixed cost thresholds redering on "Age" data
    mixed_costs_threshold(thresholds_ages,                     
                            mixed_costs_ages, 
                            best_thresholds_ages, 
                            bin_size_ages, 
                            'Age')
    # call for histgram redering on Height data
    hist(data_heights, bin_size_heights, 'Height')
    
    # call for mixed cost thresholds redering on Height data
    mixed_costs_threshold(thresholds_heights,                   
                            mixed_costs_heights, 
                            best_thresholds_heights, 
                            bin_size_heights, 
                            'Height')

    # call for ROC curve redering on Height data
    ROC_curve(TPR_FPR_tuples, roc_points_with_lowest_cost)



if __name__ == "__main__":
    main()