import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def mystery_data_sec():
    
    print("Section 1: Mystery Data:")

    # Read raw data as pandas' DataFrame
    df_mystery_data = pd.read_csv('Mystery_Data_2195.csv') 
    
    # Get column data as pandas Series
    s_mystery_data = df_mystery_data[
                        df_mystery_data.columns[0]]

    print("(before) Average =            " + str(s_mystery_data.mean().astype(float)))
    print("(before) Standard Deviation = " + str(s_mystery_data.std().astype(float))) 
    # Remove the last value from the data
    s_removed_last_val = s_mystery_data[:len(s_mystery_data)]

    print("(after) Average =             " + str(s_removed_last_val
                                    .mean().astype(float)))
    print("(after) Standard Deviation =  " + str(s_removed_last_val
                                        .std().astype(float))) 




def _quantize_binning(s_raw, str_feature):
    s_bins = pd.Series(range(
                            int(s_raw.min()), 
                            int(s_raw.max()) + 4, 
                            2))

    s_quantized = pd.cut(s_raw, s_bins, right = False) \
                                    .value_counts() \
                                    .sort_index() 

    s_quantized_value = [int(quantized_count) for quantized_count in s_quantized]
    s_quantized_value.append(0)

    # Create a DataFrame that has two columns: Height and Counts
    df_quantized_counts = pd.DataFrame(columns = [str_feature, 'Counts'])
    df_quantized_counts[str_feature] = s_bins
    df_quantized_counts['Counts'] = s_quantized_value
    return df_quantized_counts

def _fraction(s_counts, inx_counts):
    return s_counts[:inx_counts + 1].sum()/s_counts.sum()

def _variance(s_counts, inx_counts, left):
    if left:
        # return s_counts[:inx_counts].var()
        return np.var(s_counts[:inx_counts])
    else:
        # return s_counts[inx_counts:].var()
        return np.var(s_counts[inx_counts:])

def _var(s_counts, inx_counts, left):
    length = s_counts.size
    sum = 0
    if left:
        size = s_counts[:inx_counts].size
        mean = s_counts[:inx_counts].sum() /size
        for i in range(0, inx_counts):
            sum += (s_counts[i] - mean) ** 2
        return sum/size
    else:
        size = s_counts[inx_counts:].size
        mean = s_counts[inx_counts:].sum() /size
        for i in range(inx_counts, size):
            sum += (s_counts[i] - mean) ** 2
        return sum/size



def _Otsu(df_snowfolks_data_quantized):
    # print(df_snowfolks_data_quantized['Age'])
    s_height_counts = df_snowfolks_data_quantized['Counts']
    # print(s_height_counts.sum()) # index + counts
    
    best_mixed_variance = np.inf
    best_threshold = 1;

    for inx_count in range(2, len(s_height_counts) - 1):
        wt_left = _fraction(s_height_counts, inx_count)
        wt_right = 1.0 - wt_left
        wt_var_left = _variance(s_height_counts, inx_count, True)
        wt_var_right = _variance(s_height_counts, inx_count, False)
        mixed_variance = wt_left * wt_var_left \
                        + wt_right * wt_var_right
        # print(str(s_height_counts[inx_count]) + "  " + str(wt_left*100) + "%  " + "  " + str(wt_right*100) + "%  " + str(wt_var_left)+ "  " + str(wt_var_right))
        if (mixed_variance < best_mixed_variance):
            best_mixed_variance = mixed_variance
            best_threshold = inx_count
        print(mixed_variance)
        print(inx_count)


    print('best mixed var = ' + str(best_mixed_variance))
    print('best threshold = ' + str(best_threshold))

    return best_threshold

def _Otsu_reg(df_snowfolks_data_quantized, str_feature):
    # print(df_snowfolks_data_quantized['Age'])
    s_height_counts = df_snowfolks_data_quantized['Counts']
    s_feature =  df_snowfolks_data_quantized[str_feature]
    print(s_feature)
    # print(s_height_counts.sum()) # index + counts
    
    best_cost = np.inf
    best_threshold = 1;

    for inx_count in range(2, len(s_height_counts) - 1):
        wt_left = _fraction(s_height_counts, inx_count)
        wt_right = 1.0 - wt_left
        # wt_var_left = _variance(s_height_counts, inx_count, True)
        # wt_var_right = _variance(s_height_counts, inx_count, False)
        wt_var_left = _variance(s_height_counts, inx_count, True)
        wt_var_right = _variance(s_height_counts, inx_count, False)
        mixed_variance = wt_left * wt_var_left \
                        + wt_right * wt_var_right
        
        left_sum = s_height_counts[:inx_count + 1].sum()
        right_sum = s_height_counts[inx_count + 1:].sum()
        mix_cost = mixed_variance + abs(left_sum - right_sum)/25

        print(" Index " + str(inx_count) \
            + " Age " + str(s_feature[inx_count]) \
            + ": Mixed var: " + str(mixed_variance) \
            + " abs:  " + str(abs(left_sum - right_sum)/25)\
            + " Cost: " + str(mix_cost))

        if (mix_cost < best_cost):
            best_cost = mix_cost
            best_threshold = inx_count
    
    print('best cost = ' + str(best_cost))
    print('best threshold = ' + str(s_feature[best_threshold]))
    return s_feature[best_threshold]

# def _plot_histograme_by_series(df_quantized_binned_data, str_feature, point):
#     list_quantized_name = df_quantized_binned_data[str_feature]
#     list_quantized_value = df_quantized_binned_data['Counts']
#     plt.bar(list_quantized_name, list_quantized_value)
#     plt.vlines([point], colors = 'r')
#     # plt.plot(list_quantized_name, list_quantized_value)
#     plt.show()

def snowfolks_species_clustering_sec(str_feature):
    df_snowfolks_data_raw = pd.read_csv('Abominable_Data_For_Clustering__v44.csv')
    
    
    s_snowfolks_data_raw= np.floor(df_snowfolks_data_raw[str_feature] / 2) * 2   
    df_snowfolks_data_quantized = _quantize_binning(s_snowfolks_data_raw, str_feature)
    print(df_snowfolks_data_quantized)
    best = _Otsu_reg(df_snowfolks_data_quantized, str_feature)

    list_quantized_name = df_snowfolks_data_quantized[str_feature]
    list_quantized_value = df_snowfolks_data_quantized['Counts']
    plt.bar(list_quantized_name, list_quantized_value)
    plt.vlines([best], list_quantized_value.min(), list_quantized_value.max(), colors = 'r')
    # plt.plot(list_quantized_name, list_quantized_value)
    plt.show()
    
    # _plot_histograme_by_series(df_snowfolks_data_quantized, str_feature, best)
    
    # print(s_snowfolks_data_quantized)
    # print(s_snowfolks_data_quantized.sum())
    # print(s_snowfolks_data_quantized.describe(include = 'category'))
    # print(s_snowfolks_data_quantized.index)




def main():
    # mystery_data_sec()
    # snowfolks_species_clustering_sec('Age')
    snowfolks_species_clustering_sec('Ht')


if __name__ == "__main__":
    main()