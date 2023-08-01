
# IMPORTS

# Data Handling
import pandas as pd
import numpy as np
import scipy.stats as stats

# Visualization
import matplotlib.pyplot as plt

# Statistical Analysis
import statistics
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, ttest_ind

######################################################################################################################################################################


# CONFIGURATION VARIABLES - SET BEFORE RUNNING THE PROGRAM

# where the results will be stored. 
# in order to work properly, please create the directory first before hand, and make sure there are subdirectories called "tables" and "plots" within 
# or else you will get an error. In other words, the code will not make those directories on its own. Provided is a sample directory to show you how
# to set up before running. 
save_prefix = './sample/'

######################################################################################################################################################################

# FUNCTIONS - All methods have docstrings, please refer to them if you have any confusion. 

def plot_histogram(values, filename):
    """Plots a generic histogram and saves image to the plots folder under the given filename, but is hard-coded for gpa difference analysis.

    Args:
        values (List): Contains a list of values
        filename (String): Filepath to write results
    """
    plt.figure()
    plt.hist(values, bins=20, rwidth = 0.9)
    plt.xlabel('Prior GPA Differences')
    plt.ylabel('Count')
    plt.title('Prior GPA Difference Distribution')
    plt.savefig(filename)  
    plt.close()
    
def plot_density(dist1, dist2, dist1_name, dist2_name, variable, primary_color, secondary_color, title, xlabel, dist_count=2):
    """Plots density functions of two distributions. Plot saved to the plots folder under the given title. 

    Args:
        dist1 (DataFrame): A distribution of data.
        dist2 (DataFrame): Another distribution of data.
        dist1_name (String): The name of distribution 1.
        dist2_name (String): The name of distribution 2.
        variable (String): The variable (as a key in the dataframe) to compare amongst distribution 1 and distribution 2.
        primary_color (String): The color to visualize distribution 1 as.
        secondary_color (String): The color to visualize distribution 2 as.
        title (String): Title to describe the plot (and the name of the file to save as - no extensions)
        xlabel (String): Description of the variable argument.
        dist_count (int, optional): Originally implemented for more than 2 distributions, but not finished. Keep the default, which defaults to 2.
    """
    
    x = []
    colors = []
    labels = []
    labels.append(dist1_name + ' (' + str(len(dist1[variable])) + ')')
    colors.append(primary_color)
    x.append(dist1[variable])
    if (dist_count == 2):
        x.append(dist2[variable])
    if (dist_count == 2):
        colors.append(secondary_color)
    if (dist_count == 2):
        labels.append(dist2_name + ' (' + str(len(dist2[variable])) + ')')
        
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.hist(x, 20, rwidth = 0.85, density = True,  
         histtype ='bar', 
         color = colors, 
         label = labels,
         edgecolor='black', linewidth=1.0) 
    
    plt.legend(prop ={'size': 10})   
    plt.savefig(save_prefix + 'plots/' + title + '.png')
    plt.close()
    
def categorical_to_numerical(_data):
    
    """Converts all columns of a dataframe to be numerical instead of categorical. Useful for converting string type columns to int. 
    Is necessary for parts of the matching algorithm.

    Args:
        _data (DataFrame): A DataFrame object that you'd like to be all numerical
        
    Returns:
        2-tuple (DataFrame, map): Returns a converted dataframe which is only numerical (by enumeration). 
        A map is returned as the second object which maps numerical data back to its categorical counterpart.
    """
    data = _data.copy()
    name_map = {}
    real_map = data.applymap(np.isreal)
    for col_name in data:
        col = data[col_name]
        if not np.all(real_map[col_name]):
            name_map[col_name] = {}
            types = {}
            counter = 0
            for idx, value in enumerate(col):
                if not value in types.keys():
                    types[value] = counter
                    name_map[col_name][counter] = value
                    counter += 1
                
                data.at[idx, col_name] = types[value]
                
    return data, name_map

def print_map(_map, _key=None):
    """Prints a map for debugging purposes. Supports printing values for a particular optional key.

    Args:
        _map (map): A python map with keys the same type as _key.
        _key (String, optional): A key for printing out values of a particular key. Prints all keys if None. Defaults to None.
    """
    if (_key == None):
        for key in _map.keys():
            print("{}: {}".format(key, _map[key]))
            print()
    else:
        print("{}: {}".format(_key, _map[_key]))

def calculate_average_treatment_effect(df, key):
    """Calculates Average Treatment Effect (ATE) of a dataset. Hard-coded for particular group names. Used as a helper for the main ATE method.

    Args:
        df (DataFrame): Must have the columns: "group", "GPA_Post", and "GPA_Pre". "group" must be split between either 'ERSP' or 'Control'. 
        If you don't have "GPA_Post" or "GPA_Pre" in the original dataset, "GPA_Post" and "GPA_Pre" should be added to the dataset generated by match.py
        key (String): Sometimes only a subgroup is analyzed here, key should be the name of the subgroup. 

    Returns:
        float: Returns the average treatment effect between the Control and Treated groups.
    """
    treated = df[df.group == 'ERSP'].copy()
    control = df[df.group == 'Control'].copy()
    print(key)
    # print("treated: ", (treated['GPA_Post'] - treated['GPA_Pre']).mean())
    # print("control: ", (control['GPA_Post'] - control['GPA_Pre']).mean())
    print("treated: ", (treated['GPA_Post'] - treated['GPA_Pre']).mean())
    print("control: ", (control['GPA_Post'] - control['GPA_Pre']).mean())
    # ate = (treated['GPA_Post'] - treated['GPA_Pre']).mean() - (control['GPA_Post'] - control['GPA_Pre']).mean()
    ate = (treated['GPA_Post'] - treated['GPA_Pre']).mean() - (control['GPA_Post'] - control['GPA_Pre']).mean()
    return ate
    
def get_demographic_stats(df, category_map):
    """Generates a table of demographic stats (percentages) with hard-coded filenames. 
    I don't remember why I decided to make it work on the converted-to-numerical dataset,
    but that is what this function requires. 

    Args:
        df (DataFrame): An all-numerical dataset
        category_map (map): A map describing corresponding enums with their categories.
        
        Can generate both arguments from the categorical_to_numerical method.
    """
    treated_stats = np.zeros(shape=(len(category_map['SEX']), len(category_map['eth_grp'])))
    control_stats = np.zeros(shape=(len(category_map['SEX']), len(category_map['eth_grp'])))
    
    for sex_id in category_map['SEX'].keys():
        for eth_id in category_map['eth_grp'].keys():
            treated_stats[sex_id,eth_id] = len(df[(df.SEX == sex_id) & (df.eth_grp == eth_id) & (df.group == 0)])
            control_stats[sex_id,eth_id] = len(df[(df.SEX == sex_id) & (df.eth_grp == eth_id) & (df.group == 1)])
            
    num_first_gen_ersp = len(df[(df.group == 0) & (df.first_generation == 1)]) / len(df[df.group == 0])
    num_first_gen_control = len(df[(df.group == 1) & (df.first_generation == 1)]) / len(df[df.group == 1])
    print("first gen ersp%: {}".format(num_first_gen_ersp))
    print("first gen control%: {}".format(num_first_gen_control))
    treated_n = sum(sum(treated_stats))
    control_n = sum(sum(control_stats))
    probability_transform_treated = lambda x: (x / treated_n) * 100
    probability_transform_control = lambda x: (x / control_n) * 100
    treated_stats_perc = probability_transform_treated(treated_stats)
    control_stats_perc = probability_transform_control(control_stats)
    
    np.set_printoptions(precision=5)
    print("treated counts:")
    print(treated_stats)
    print("control stats")
    print(control_stats)
    result_treated = {}
    result_control = {}
    for eth in range(len(category_map['eth_grp'])):
        result_treated[category_map['eth_grp'][eth]] = {}
        result_control[category_map['eth_grp'][eth]] = {}
        for sex in range(len(category_map['SEX'])):
            result_treated[category_map['eth_grp'][eth]][category_map['SEX'][sex]] = treated_stats_perc[sex,eth]
            result_control[category_map['eth_grp'][eth]][category_map['SEX'][sex]] = control_stats_perc[sex,eth]
            
    pd.DataFrame(result_treated).to_csv(save_prefix + 'tables/ERSP_demographics.csv',float_format='%.5f')
    pd.DataFrame(result_control).to_csv(save_prefix + 'tables/Control_Demographics.csv',float_format='%.5f')
        
def get_difference_count(_df, propensity_vars):
    """Counts the number of differences between matched variables and constructs a table with these counts. 
    For example, if two matched students differ on an ethnicity variable, this counts as a variable difference,
    which contributes to the cell for ethnicity.

    Args:
        _df (DataFrame): An all-numerical dataframe. It is essential that this is all numerical, since the way the algorithm checks if two variables are the same is by subtracting 
        the columns and checking if the result is 0. 
        propensity_vars (List): A list of variables of interest. 
    """
    vars = propensity_vars
    df = _df[vars]
    differences = {}
    subtracted = pd.DataFrame()
    gpa_difference = []
    raw_gpa_difference = []
    
    gpa_diff_count = 0
    gpa_diff_count_total = 0
    for i in range(0, len(df), 2):
        row1 = df.iloc[i]
        row2 = df.iloc[i+1]
        new_row = row1 - row2
        subtracted = pd.concat([subtracted, new_row], axis=1, ignore_index=True)
        
        # gpa_row1 = _df.at[i,'F2GPA'] if (_df.at[i,'YR2 Match'] == 1) else _df.iloc[i]['F3GPA']
        # gpa_row2 = _df.at[i+1,'F2GPA'] if (_df.at[i+1,'YR2 Match'] == 1) else _df.iloc[i+1]['F3GPA']
        gpa_row1 = row1['GPA_Pre']
        gpa_row2 = row2['GPA_Pre']
        gpa_difference.append(abs(gpa_row1 - gpa_row2))
        raw_gpa_difference.append(gpa_row1 - gpa_row2)
        
        if (abs(gpa_row1 - gpa_row2) < 0.1):
            gpa_diff_count += 1
        
        gpa_diff_count_total += 1
        
    print("percent of ersp students who matched with a control student with a difference of 0.1 or less gpa: {}".format(gpa_diff_count / gpa_diff_count_total))

    average_absolute_difference = sum(gpa_difference) / len(gpa_difference)
    gpa_difference_std_dev = statistics.pstdev(gpa_difference)
    plot_histogram(raw_gpa_difference, save_prefix + 'plots/GPA_Difference.png')
    subtracted = subtracted.T.reset_index(drop=True)
    for col in vars:
        differences[col] = len(subtracted[subtracted[col] != 0])

    gpa_stats = {
        'gpa average absolute difference': average_absolute_difference,
        'gpa difference standard deviation': gpa_difference_std_dev
    }
    pd.DataFrame(gpa_stats.items(), columns=['Stat', 'Value']).to_csv(save_prefix + 'tables/gpa_difference_stats.csv', index=False, float_format='%.2f')
    pd.DataFrame(differences.items(), columns=['Variable', 'Difference']).to_csv(save_prefix + 'tables/variable_difference_count.csv', index=False)
    
def get_similarity_stats(df, propensity_vars):
    """DO NOT USE   :O   ... This method finds.... a type of similarity? What is important is that I ended up not using this method, and instead used a percentage to calculate similarity. 
    For example, if all the students matached perfectly on admit cohort, that variable would have 100% similarity. If 7/10 groups matched perfectly, this was 70%. 

    Args:
        df (DataFrame): The entire distribution
        propensity_vars (List): Variable keys of interest.
    """
    vars = propensity_vars
    treated = df[df['group'] == 0]
    control = df[df['group'] == 1]
    similarity_scores = {}
    for var in vars:
        treated_list = [x for x in treated[var]]
        control_list = [x for x in control[var]]
        p_value = stats.ttest_ind(treated_list, control_list)[1]
        similarity_scores[var] = p_value
        
    pd.DataFrame(similarity_scores.items(), columns=['Variable', 'Similarity']).to_csv(save_prefix + 'tables/Similarity_Scores.csv', index=False, float_format='%.5f')
    
def get_average_treatment_effect(df):
    """The main ATE method. It calculates Average Treatment Effect for multiple subgroups of interest (hard-coded). There is much abstraction here.
    Make sure your input DataFrame follows the specifications below. Saves a table of results to a hard-coded filepath. 

    Args:
        df (DataFrame): A dataframe with the columns: "first_generation", "URM", and "SEX". It calculates ATE, p-score, and z-score. 
    """
    subgroups_of_interest = {
        'all': df, 
        'first_generation':df[df.first_generation == 1], 
        'not first_generation':df[df.first_generation == 0],
        'URM':df[df.URM == 1], 
        'Not URM':df[df.URM == 0],
        'Male':df[df.SEX == 'Male'],
        'Female':df[df.SEX == 'Female'] 
    }
    
    ate_scores = {
        'all':[0,0,0,0],
        'first_generation':[0,0,0,0],
        'not first_generation':[0,0,0,0],
        'URM':[0,0,0,0], 
        'Not URM':[0,0,0,0],
        'Male':[0,0,0,0],
        'Female':[0,0,0,0]
    }

    
    for key in subgroups_of_interest.keys():
        ate_scores[key][0] = calculate_average_treatment_effect(subgroups_of_interest[key], key)
        
        treated = subgroups_of_interest[key][subgroups_of_interest[key].group == 'ERSP']
        control = subgroups_of_interest[key][subgroups_of_interest[key].group == 'Control']
    
        #scipy 
        #pooled if variance roughly equal
        # leave out ddof
        # alt: larger
        # z_score, p_value = ws.ztest(treated['GPA_Diff'], control['GPA_Diff'], 
        #                             value=0, alternative='larger', 
        #                             usevar='pooled', ddof=1)
        z_score, p_value = ttest_ind(treated['GPA_Diff'], control['GPA_Diff'], equal_var=True, alternative='greater')
        # z_score, p_value = ttest_ind(treated['GPA_Diff'], control['GPA_Diff'], equal_var=True)
        ate_scores[key][1] = p_value
        ate_scores[key][2] = z_score
        ate_scores[key][3] = len(treated['GPA_Diff'])
        
        
    results = pd.DataFrame(columns = ['Subgroup', 'ATE', 'p-value', 'z-score', 'n'])
    
    keys = ate_scores.keys()
    results['Subgroup'] = keys
    results['ATE'] = [ate_scores[x][0] for x in keys]
    results['p-value'] = [ate_scores[x][1] for x in keys]
    results['z-score'] = [ate_scores[x][2] for x in keys]
    results['n'] = [ate_scores[x][3] for x in keys]

    results.to_csv(save_prefix + 'tables/ate_stats.csv', index=False)
        
def calculate_retention_rates(df,test_name):
    """Calculates retention rates of a subgroup. This is used as a helper function to the main retention function. 

    Args:
        df (DataFrame): An already pre-filtered dataframe which only includes the subgroup specified by test_name. 
        test_name (String): Subgroup of interest, which has already filtered df.

    Returns:
        float: a retention percentage for the given subgroup.
    """
    print("calculating retention for {}:".format(test_name))
    treated = df[df.group == 'ERSP']
    control = df[df.group == 'Control']
    # print("treated")
    # print(treated)
    # print()
    # print("control")
    # print(control)
    computing_majors = ['CMPSC', 'CMPEN', 'STSDS', 'CPTCS']
    
    treated_n = len(df[df.group == 'ERSP'])
    control_n = len(df[df.group == 'Control'])
    print("treated_n: {}".format(treated_n))
    print("control_n: {}".format(control_n))

    # TODO: Or Last Major2
    treated_major_retention = len(treated[treated.Last_Major1.isin(computing_majors) | treated.Last_Major2.isin(computing_majors)])
    control_major_retention = len(control[control.Last_Major1.isin(computing_majors) | control.Last_Major2.isin(computing_majors)])

    retentions = [control_major_retention / control_n, treated_major_retention / treated_n]
    retention_df = pd.DataFrame({'Group': ['Control', 'Treated'],
                   'Retention': retentions,
                   'Total': [control_n, treated_n]})
    
    nobs = retention_df['Total']
    successes = retention_df['Retention'] * retention_df['Total']

    # Conduct the proportions z-test
    stat, pval = proportions_ztest(successes, nobs)
    # Apply Holm's method to adjust the p-values
    pvals_adjusted = multipletests(pval, method='holm')[1][0]


    return [retentions[1], treated_n, retentions[0], control_n, stat, pval, pvals_adjusted]

def get_retention_stats(df, path):
    """The main retention method. Saves retention results to a specified path. Calculates retention,
    test-statistic values, p-values, and holm-adjusted p-values. 

    Args:
        df (DataFrame): The entire unfiltered dataframe. 
        path (String): Path to save the table. (For file cleanliness, please provide the path with the table prefix, 
        as I do not do that automatically for this method - sorry. See the main method for an example of how I use these methods). 
    """
    subgroups_of_interest = {
        'Full Sample': df, 
        'First Gen':df[df.first_generation == 1], 
        'Not First Gen':df[df.first_generation == 0],
        'URM':df[df.URM == 1], 
        'Non-URM':df[df.URM == 0],
        'Male':df[df.SEX == 'Male'],
        'Female':df[df.SEX == 'Female'] 
    }
    
    index = ['Full Sample', 'Female', 'Male', 'URM', 'Non-URM', 'First Gen', 'Not First Gen']
    # index = ['Full Sample', 'Female', 'Male', 'URM', 'Non-URM']
    # index = ['Full Sample', 'Female', 'Male']
    
    retention_stats = {
        'ERSP': [],
        'ERSP n': [],
        'Control': [],
        'Control n': [],
        'Test Statistic': [],
        'p-value': [],
        'Holm Adjustment': []
    }
    
    for idx, key in enumerate(index):
        
        results = calculate_retention_rates(subgroups_of_interest[key], key)
        print("results")
        print(results)
        retention_stats['ERSP'].append(results[0])
        retention_stats['ERSP n'].append(results[1])
        retention_stats['Control'].append(results[2])
        retention_stats['Control n'].append(results[3])
        retention_stats['Test Statistic'].append(results[4])
        retention_stats['p-value'].append(results[5])
        # retention_stats['Holm Adjustment'].append(results[6])
    print("p-value!")
    print(retention_stats['p-value'])
        
    adjusted = multipletests(retention_stats['p-value'], method='holm')[1]
    sort_indices = np.array(retention_stats['p-value']).argsort()
    print(sort_indices)
    print(adjusted)
    retention_stats['Holm Adjustment'] = adjusted[sort_indices]
    print(retention_stats)
    pd.DataFrame(retention_stats, index=index).to_csv(path, float_format='%.3f')
    
def calculate_mann_whitney_u(dist1, dist2, variable):
    """A helper function for the main mann whitney u method. Calculates the U statistic, medians, and p-value between two distributions. 

    Args:
        dist1 (DataFrame): A distribution that includes the variable argument as a column.
        dist2 (DataFrame): Another distribution that includes the variable argument as a column. 
        variable (String): A variable of interest. 

    Returns:
        result (map): A mapping of the different results to their values. Includes U-stat, medians, and p-value. 
    """
    u_stat, pval = mannwhitneyu(dist1[variable], dist2[variable])
    
    result = {
        'U': u_stat,
        'median1': dist1[variable].median(),
        'median2': dist2[variable].median(),
        'p-value': pval,
    }
    return result
    
def get_mann_whitney_u_stats(_df):
    """The main mann whitney u method. It finds a the u-stat, medians, and p-value of several different hard-coded subgroups of interest.  
    It will produce plots of the densities between groups, as well as a table with the stats mentioned. The plots names and paths are hard-coded,
    as well as the sub-groups of interest. 

    Args:
        _df (DataFrame): The entire unfiltered dataframe.
 
    """
    
    df = _df[_df.group == 'ERSP']
    df_control = _df[_df.group == 'Control']
    
    df_fg = df[df.first_generation == 1].copy()
    df_not_fg = df[df.first_generation == 0].copy()
    df_urm = df[df.URM == 1].copy()
    df_not_urm = df[df.URM == 0].copy()
    df_male = df[df.SEX == 'Male'].copy()
    df_female = df[df.SEX == 'Female'].copy()
    
    control_df_fg = df_control[df_control.first_generation == 1].copy()
    control_df_not_fg = df_control[df_control.first_generation == 0].copy()
    control_df_urm = df_control[df_control.URM == 1].copy()
    control_df_not_urm = df_control[df_control.URM == 0].copy()
    control_df_male = df_control[df_control.SEX == 'Male'].copy()
    control_df_female = df_control[df_control.SEX == 'Female'].copy()
    
    # Plots ERSP
    plot_density(dist1 = df, dist2 = None, dist1_name="ALL", dist2_name=None, variable='GPA_Pre', primary_color='blue', secondary_color=None, title="Prior GPA (ERSP)", xlabel='GPA prior to ERSP',dist_count=1)
    plot_density(dist1 = df, dist2 = None, dist1_name="ALL", dist2_name=None, variable='GPA_Post', primary_color='blue', secondary_color=None, title="Post GPA (ERSP)", xlabel='GPA Post ERSP',dist_count=1)
    plot_density(dist1 = df, dist2 = None, dist1_name="ALL", dist2_name=None, variable='GPA_Diff', primary_color='blue', secondary_color=None, title="GPA Difference (ERSP)", xlabel='GPA Difference (Post - Prior to ERSP)',dist_count=1)
    
    plot_density(dist1 = df_fg, dist2 = df_not_fg, dist1_name="first_generation", dist2_name="not first generation", variable='GPA_Pre', primary_color='blue', secondary_color='lightblue', title="Prior GPA by First-Generation Flag (ERSP)", xlabel='GPA prior to ERSP')
    plot_density(dist1 = df_fg, dist2 = df_not_fg, dist1_name="first generation", dist2_name="not first generation", variable='GPA_Post', primary_color='blue', secondary_color='lightblue', title="Post GPA by First-Generation Flag (ERSP)", xlabel='GPA Post ERSP')
    plot_density(dist1 = df_fg, dist2 = df_not_fg, dist1_name="first generation", dist2_name="not first generation", variable='GPA_Diff', primary_color='blue', secondary_color='lightblue', title="GPA Difference by First-Generation Flag (ERSP)", xlabel='GPA Difference (Post - Prior to ERSP)')
    plot_density(dist1 = df_urm, dist2 = df_not_urm, dist1_name="urm", dist2_name="not urm", variable='GPA_Pre', primary_color='purple', secondary_color='thistle', title="Prior GPA by URM Flag (ERSP)", xlabel='GPA Prior to ERSP')
    plot_density(dist1 = df_urm, dist2 = df_not_urm, dist1_name="urm", dist2_name="not urm", variable='GPA_Post', primary_color='purple', secondary_color='thistle', title="Post GPA by URM Flag (ERSP)", xlabel='GPA Post ERSP')
    plot_density(dist1 = df_urm, dist2 = df_not_urm, dist1_name="urm", dist2_name="not urm", variable='GPA_Diff', primary_color='purple', secondary_color='thistle', title="GPA Difference by URM Flag (ERSP)", xlabel='GPA Difference (Post - Prior to ERSP)')
    plot_density(dist1 = df_male, dist2 = df_female, dist1_name="male", dist2_name="female", variable='GPA_Pre', primary_color='green', secondary_color='lightgreen', title="Prior GPA by Gender (ERSP)", xlabel='GPA Prior to ERSP')
    plot_density(dist1 = df_male, dist2 = df_female, dist1_name="male", dist2_name="female", variable='GPA_Post', primary_color='green', secondary_color='lightgreen', title="Post GPA by Gender (ERSP)", xlabel='GPA Post ERSP')
    plot_density(dist1 = df_male, dist2 = df_female, dist1_name="male", dist2_name="female", variable='GPA_Diff', primary_color='green', secondary_color='lightgreen', title="GPA Difference by Gender (ERSP)", xlabel='GPA Difference (Post - Prior to ERSP)')

    # Plots Control
    plot_density(dist1= df_control, dist2=None, dist1_name='ALL', dist2_name=None, variable='GPA_Pre', primary_color='blue', secondary_color='lightblue', title='Prior GPA (Control)', xlabel='GPA prior to ERSP',dist_count=1)
    plot_density(dist1= df_control, dist2=None, dist1_name='ALL', dist2_name=None, variable='GPA_Post', primary_color='blue', secondary_color='lightblue', title='Post GPA (Control)', xlabel='GPA Post ERSP',dist_count=1)
    plot_density(dist1= df_control, dist2=None, dist1_name='ALL', dist2_name=None, variable='GPA_Diff', primary_color='blue', secondary_color='lightblue', title='GPA Difference (Control) ', xlabel='GPA Difference (Post - Prior to ERSP)',dist_count=1)
    
    plot_density(dist1= df_control[df_control.first_generation == 1].copy(), dist2=df_control[df_control.first_generation == 0].copy(), dist1_name='first generation', dist2_name='not first generation', variable='GPA_Pre', primary_color='blue', secondary_color='lightblue', title='Prior GPA by first generation flag (Control)', xlabel='GPA prior to ERSP')
    plot_density(dist1= df_control[df_control.first_generation == 1].copy(), dist2=df_control[df_control.first_generation == 0].copy(), dist1_name='first generation', dist2_name='not first generation', variable='GPA_Post', primary_color='blue', secondary_color='lightblue', title='Post GPA by first generation flag (Control)', xlabel='GPA Post ERSP')
    plot_density(dist1= df_control[df_control.first_generation == 1].copy(), dist2=df_control[df_control.first_generation == 0].copy(), dist1_name='first generation', dist2_name='not first generation', variable='GPA_Diff', primary_color='blue', secondary_color='lightblue', title='GPA Difference by first generation flag (Control) ', xlabel='GPA Difference (Post - Prior to ERSP)')
    plot_density(dist1= df_control[df_control.URM == 1].copy(), dist2=df_control[df_control.URM == 0].copy(), dist1_name='urm', dist2_name='not urm', variable='GPA_Pre', primary_color='purple', secondary_color='thistle', title='Prior GPA by URM Flag (Control)', xlabel='GPA prior to ERSP')
    plot_density(dist1= df_control[df_control.URM == 1].copy(), dist2=df_control[df_control.URM == 0].copy(), dist1_name='urm', dist2_name='not urm', variable='GPA_Post', primary_color='purple', secondary_color='thistle', title='Post GPA by URM Flag (Control)', xlabel='GPA Post ERSP')
    plot_density(dist1= df_control[df_control.URM == 1].copy(), dist2=df_control[df_control.URM == 0].copy(), dist1_name='urm', dist2_name='not urm', variable='GPA_Diff', primary_color='purple', secondary_color='thistle', title='GPA Difference by URM Flag (Control) ', xlabel='GPA Difference (Post - Prior to ERSP)')
    plot_density(dist1= df_control[df_control.SEX == 'Male'].copy(), dist2=df_control[df_control.SEX == 'Female'].copy(), dist1_name='male', dist2_name='female', variable='GPA_Pre', primary_color='green', secondary_color='lightgreen', title='Prior GPA by Gender (Control)', xlabel='GPA prior to ERSP')
    plot_density(dist1= df_control[df_control.SEX == 'Male'].copy(), dist2=df_control[df_control.SEX == 'Female'].copy(), dist1_name='male', dist2_name='female', variable='GPA_Post', primary_color='green', secondary_color='lightgreen', title='Post GPA by Gender (Control)', xlabel='GPA Post ERSP')
    plot_density(dist1= df_control[df_control.SEX == 'Male'].copy(), dist2=df_control[df_control.SEX == 'Female'].copy(), dist1_name='male', dist2_name='female', variable='GPA_Diff', primary_color='green', secondary_color='lightgreen', title='GPA Difference by Gender (Control) ', xlabel='GPA Difference (Post - Prior to ERSP)')
    
    # Plots all
    plot_density(dist1= _df, dist2=None, dist1_name='ALL', dist2_name=None, variable='GPA_Pre', primary_color='blue', secondary_color='lightblue', title='Prior GPA (ALL)', xlabel='GPA prior to ERSP',dist_count=1)
    plot_density(dist1= _df, dist2=None, dist1_name='ALL', dist2_name=None, variable='GPA_Post', primary_color='blue', secondary_color='lightblue', title='Post GPA (ALL)', xlabel='GPA Post ERSP',dist_count=1)
    plot_density(dist1= _df, dist2=None, dist1_name='ALL', dist2_name=None, variable='GPA_Diff', primary_color='blue', secondary_color='lightblue', title='GPA Difference (ALL) ', xlabel='GPA Difference (Post - Prior to ERSP)',dist_count=1)
    
    plot_density(dist1= _df[_df.first_generation == 1].copy(), dist2=_df[_df.first_generation == 0].copy(), dist1_name='first generation', dist2_name='not first generation', variable='GPA_Pre', primary_color='blue', secondary_color='lightblue', title='Prior GPA by first generation flag (ALL)', xlabel='GPA prior to ERSP')
    plot_density(dist1= _df[_df.first_generation == 1].copy(), dist2=_df[_df.first_generation == 0].copy(), dist1_name='first generation', dist2_name='not first generation', variable='GPA_Post', primary_color='blue', secondary_color='lightblue', title='Post GPA by first generation flag (ALL)', xlabel='GPA Post ERSP')
    plot_density(dist1= _df[_df.first_generation == 1].copy(), dist2=_df[_df.first_generation == 0].copy(), dist1_name='first generation', dist2_name='not first generation', variable='GPA_Diff', primary_color='blue', secondary_color='lightblue', title='GPA Difference by first generation flag (ALL) ', xlabel='GPA Difference (Post - Prior to ERSP)')
    plot_density(dist1= _df[_df.URM == 1].copy(), dist2=_df[_df.URM == 0].copy(), dist1_name='urm', dist2_name='not urm', variable='GPA_Pre', primary_color='purple', secondary_color='thistle', title='Prior GPA by URM Flag (ALL)', xlabel='GPA prior to ERSP')
    plot_density(dist1= _df[_df.URM == 1].copy(), dist2=_df[_df.URM == 0].copy(), dist1_name='urm', dist2_name='not urm', variable='GPA_Post', primary_color='purple', secondary_color='thistle', title='Post GPA by URM Flag (ALL)', xlabel='GPA Post ERSP')
    plot_density(dist1= _df[_df.URM == 1].copy(), dist2=_df[_df.URM == 0].copy(), dist1_name='urm', dist2_name='not urm', variable='GPA_Diff', primary_color='purple', secondary_color='thistle', title='GPA Difference by URM Flag (ALL) ', xlabel='GPA Difference (Post - Prior to ERSP)')
    plot_density(dist1= _df[_df.SEX == 'Male'].copy(), dist2=_df[_df.SEX == 'Female'].copy(), dist1_name='male', dist2_name='female', variable='GPA_Pre', primary_color='green', secondary_color='lightgreen', title='Prior GPA by Gender (ALL)', xlabel='GPA prior to ERSP')
    plot_density(dist1= _df[_df.SEX == 'Male'].copy(), dist2=_df[_df.SEX == 'Female'].copy(), dist1_name='male', dist2_name='female', variable='GPA_Post', primary_color='green', secondary_color='lightgreen', title='Post GPA by Gender (ALL)', xlabel='GPA Post ERSP')
    plot_density(dist1= _df[_df.SEX == 'Male'].copy(), dist2=_df[_df.SEX == 'Female'].copy(), dist1_name='male', dist2_name='female', variable='GPA_Diff', primary_color='green', secondary_color='lightgreen', title='GPA Difference by Gender (ALL) ', xlabel='GPA Difference (Post - Prior to ERSP)')
    
    
    treated_subgroups_of_interest = {
        'first_gen': [df_fg, df_not_fg, save_prefix + 'tables/ERSP_mann_whitney_first_generation.csv'],
        'urm': [df_urm, df_not_urm, save_prefix + 'tables/ERSP_mann_whitney_urm.csv'],
        'sex': [df_male, df_female, save_prefix + 'tables/ERSP_mann_whitney_sex.csv']
    }
    control_subgroups_of_interest = {
        'first_gen': [control_df_fg, control_df_not_fg, save_prefix + 'tables/Control_mann_whitney_first_generation.csv'],
        'urm': [control_df_urm, control_df_not_urm, save_prefix + 'tables/Control_mann_whitney_urm.csv'],
        'sex': [control_df_male, control_df_female, save_prefix + 'tables/Control_mann_whitney_sex.csv']
    }
    
    
    
    index = ['First Generation Median', 'Non First Generation Median', 'U-Score', 'P-Value']
    dict_keys = ['median1', 'median2', 'U', 'p-value']
    for key in treated_subgroups_of_interest.keys():
        
        dist1 = treated_subgroups_of_interest[key][0]
        dist2 = treated_subgroups_of_interest[key][1]
        before = calculate_mann_whitney_u(dist1, dist2, 'GPA_Pre')
        after = calculate_mann_whitney_u(dist1, dist2, 'GPA_Post')
        diff = calculate_mann_whitney_u(dist1, dist2, 'GPA_Diff')
        
        result = {
            'before': [before[x] for x in dict_keys],
            'after': [after[x] for x in dict_keys],
            'diff': [diff[x] for x in dict_keys]
        }
        
        pd.DataFrame(result, index=index).to_csv(treated_subgroups_of_interest[key][2])
        
    for key in control_subgroups_of_interest.keys():
        
        dist1 = control_subgroups_of_interest[key][0]
        dist2 = control_subgroups_of_interest[key][1]
        before = calculate_mann_whitney_u(dist1, dist2, 'GPA_Pre')
        after = calculate_mann_whitney_u(dist1, dist2, 'GPA_Post')
        diff = calculate_mann_whitney_u(dist1, dist2, 'GPA_Diff')
        
        result = {
            'before': [before[x] for x in dict_keys],
            'after': [after[x] for x in dict_keys],
            'diff': [diff[x] for x in dict_keys]
        }
        
        pd.DataFrame(result, index=index).to_csv(control_subgroups_of_interest[key][2])
    

######################################################################################################################################################################

# MAIN - This is the main function I used for my analysis, depending on your data format, you may need to tweak hard-coded aspects in the analysis code. But, this should be
#        a good example of how to use each method, and how to set up their preconditions. 


def main():
    
    ######################################################################################################################################################################

    # STEP 1: Prep the data
    # STEP 1a: Start out with the file produced by match.py
    #         For me, this was called "MATCHED_with_csceqtr copy.csv"
    #         Pay attention to the name of the file you chose.
    df = pd.read_csv('MATCHED_with_csceqtr copy.csv')
    
    # STEP 1b: Get all the data just in case. I happened to use it to generate
    #         Retention data for the entire control group and compare to the
    #         matched control group. 
    df_all = pd.read_csv('All-data-NEW-ERSP_Full-Control-Group.csv')

    # STEP 1c: Get an all-numerical dataset, as some methods require it.
    df_numerical, category_map = categorical_to_numerical(df)
    
    # STEP 1d: Select the variables of interest for similarity and difference checks. 
    #          I used this list for getting difference counts.
    propensity_vars = ['first_generation', 'Admit_Cohort', 'Major_at_Admission', 'CS_CE_qtr', 'eth_grp', 'SEX', 'GPA_Pre']
    
    ######################################################################################################################################################################

    
    # STEP 2: Get Stats for the Matching algorithm
    # STEP 2a: Pass in numerical data and the coresponding map - see details in get_demographic_stats's docstring
    print()
    print("DEMOGRAPHICS")
    get_demographic_stats(df_numerical, category_map)
    print("\tDemographic Stats reported in '" + save_prefix + "tables/Control_Demographics.csv' and '" + save_prefix + "tables/ERSP_Demographics.csv'\n")
    
    # STEP 2b: Find the number of differences between variables of interest. Pass in a numerical dataset and variables of interest
    #          See the doc-string of get_difference_count for details.
    print("DIFFERENCE")
    get_difference_count(df_numerical, propensity_vars)
    print("\tGPA difference Stats reported in '" + save_prefix + "tables/gpa_difference_stats.csv'")
    print("\tVariable difference counts reported in '" + save_prefix + "tables/variable_difference_count.csv'")
    print("\tGPA difference plot reported in '" + save_prefix + "plots/GPA_Difference.png'\n")
    
    # STEP 2c: Find the similarity measure of the matched group:
    #          WARNING: I DID NOT USE THIS METHOD, see get_similarity_stats's doc-string for details. 
    print("SIMILARITY")
    get_similarity_stats(df_numerical, propensity_vars)
    print("\tVariable similarity scores reported in '" + save_prefix + "tables/Similarity_Scores.csv'\n")
    
    ######################################################################################################################################################################

    # STEP 3: Main Objective Analysis - ATE, Retention, and Mann Whitney U.
    # STEP 3a: Find the average treatment effect. Pass in the matched dataset. See the doc-string
    #          for get_average_treatment_effect for details.
    print("ATE")
    print(df)
    get_average_treatment_effect(df)
    print("\tAverage Treatment Effect stats reported to '" + save_prefix + "tables/ate_stats.csv'\n")
    
    # STEP 3b: Find the retention among the matched groups. Here, I am doing analysis between both
    #          the matched only group and the entire control group. See the doc-string for 
    #          get_retention_stats for details. 
    print("RETENTION")
    get_retention_stats(df, save_prefix + 'tables/Retention_Data.csv')
    get_retention_stats(df_all, save_prefix + 'tables/ALL_Retention_Data.csv')
    print("Retention Stats reported to 'Retention_Data.csv'\n")
    
    # STEP 3c: Find the Mann Whitney U stats. Pass in the entire matched dataset. See the doc-string
    #          of get_mann_whitney_u_stats for details. 
    print("MANN WHITNEY U")
    get_mann_whitney_u_stats(df)
    print("Mann Whitney Stats reported to various tables and plots in '" + save_prefix + "plots' and '" + save_prefix + "tables'\n")
    
    ######################################################################################################################################################################
    # Enjoy :D and let me know if you have any questions: email me at zglazewski@ucsb.edu
    
if __name__ == '__main__':
    main()