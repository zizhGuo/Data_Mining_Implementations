import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    df_snowfolks_data_raw = pd.read_csv('Abominable_Data_For_Clustering__v44.csv')
    s_snowfolks_data_raw= np.floor(df_snowfolks_data_raw['Ht'] / 2) * 2
    print(s_snowfolks_data_raw)
    
    # x = s_snowfolks_data_raw[s_snowfolks_data_raw > 180 ]

    df = pd.DataFrame()

    start = s_snowfolks_data_raw.min() + 2
    flag = s_snowfolks_data_raw.min() + 2
    end = s_snowfolks_data_raw.max()
    # it = range(start, end)
    
    while s_snowfolks_data_raw.empty is not True:
        x = s_snowfolks_data_raw[s_snowfolks_data_raw < flag]
        df[flag] = [x.size]
        s_snowfolks_data_raw = s_snowfolks_data_raw[s_snowfolks_data_raw >= flag]
        flag += 2

    ages = np.array([df[x][0] for x in df.columns])
    print(ages)
    cost_values = []
    best_cost = np.inf
    best_threshold = 1;

    for x in range(1, ages.size- 1):
        wt_left = np.sum(ages[:x+1])/np.sum(ages)
        wt_right = np.sum(ages[x+1:])/np.sum(ages)
        # wt_right = 1.0 - wt_left
        wt_var_left = np.var(ages[:x+1])
        wt_var_right = np.var(ages[x+1:])

        mixed_variance = wt_left * wt_var_left + wt_right * wt_var_right
        reg = abs(np.sum(ages[:x+1])-np.sum(ages[x+1:]))/25
        cost = mixed_variance + reg
        cost_values.append(mixed_variance)

        if (cost < best_cost):
            best_cost = cost
            best_threshold = x
    #     print(str(df.columns[best_threshold])+ "   " + str(cost))
    print(cost_values)
    cost_values.insert(0, 0)
    cost_values.append(0)
    # print(best_cost)
    # print(df.columns[best_threshold])
        # print(mixed_variance)

    # print(np.sum(ages[:5]))
    # print(df.min())
    
        # print(x)
        # print(df[x][0])

    # Hist
    plt.bar(df.columns, [df[x][0] for x in df.columns])
    plt.vlines([df.columns[best_threshold]], np.min(ages), np.max(ages), colors = 'r')
    plt.ylabel('# People for each bins')
    plt.xlabel('Height bins')

    # Cost
    # print(len(cost_values))
    # print(len(df.columns.counts))
    # plt.bar(df.columns, cost_values)
    # plt.plot(df.columns, cost_values)
    # plt.vlines([df.columns[best_threshold]], np.min(cost_values), np.max(cost_values), colors = 'r')
    # plt.ylabel('Mixed Cost')
    # plt.xlabel('Ht Counts bins')
    plt.show()
    # print(df.columns[108.0])
        
    # print(s_snowfolks_data_raw)
    # print(s_snowfolks_data_raw[s_snowfolks_data_raw > 200].empty)