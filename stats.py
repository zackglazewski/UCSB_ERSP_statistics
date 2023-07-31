import pandas as pd
import numpy as np
import scipy.stats as stats
import statistics
import matplotlib.pyplot as plt
import statsmodels.stats.weightstats as ws
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind


# where the results will be stored. 
# in order to work properly, please create the directory first before hand, and make sure there are subdirectories called "tables" and "plots" within 
# or else you will get an error. In other words, the code will not make those directories on its own. 
save_prefix = './sample/'

# plots gpa histogram
def plot_histogram(values, title):
    plt.figure()
    plt.hist(values, bins=20, rwidth = 0.9)
    plt.xlabel('Prior GPA Differences')
    plt.ylabel('Count')
    plt.title('Prior GPA Difference Distribution')
    plt.savefig(save_prefix + 'plots/gpa_difference_histogram.png')  
    plt.close()
    
# plots density charts
def plot_density(dist1, dist2, dist1_name, dist2_name, variable, primary_color, secondary_color, title, xlabel, dist_count=2):
    # plt.figure()
    # sns.histplot(data=dist1[variable], kde=True, stat="density", color=primary_color, alpha=0.5, label="Treated", bins=20)
    # sns.histplot(data=dist2[variable], kde=True, stat="density", color=secondary_color, alpha=0.5, label="Control", bins=20)
 
    # plt.xlabel(variable)
    # plt.ylabel("density")
    # plt.title(title)
 
    # plt.savefig('./data/' + title)
    # plt.close()
    # df = pd.concat([dist1, dist2], keys=['df1', 'df2']).copy()
    
    # plt.figure()
    # df['Distribution'] = ['Distribution 1']*len(dist1) + ['Distribution 2']*len(dist2)
    # sns.histplot(data=df, x=variable, hue='Distribution', element='step', stat='density')
    # plt.savefig('./data/' + title)
    
    # plt.close()
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
    
# same as in match.py
def categorical_to_numerical(_data):
    
    """Takes in data and columns, converts categorical data to discrete numerical values.

    Returns:
        (DataFrame, Dict): DataFrame: altered data df with numerical values, Dict: Mapping of numbers to categories for future use
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

# prints a map for debugging
def print_map(_map, _key=None):
    if (_key == None):
        for key in _map.keys():
            print("{}: {}".format(key, _map[key]))
            print()
    else:
        print("{}: {}".format(_key, _map[_key]))

# calculates averate treamtment effect, only need to pass in dataset, lots of abstraction as long as all the right variables exist in the df.
def calculate_average_treatment_effect(df, key):
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
    
# gets demographic percentages, like in match.py
def get_demographic_stats(df, category_map):
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
        
# counts how many differences exist among each matching variable. For example, if two matched students differ on a ethnicity variable, this counts as a variable difference.
def get_difference_count(_df, propensity_vars):
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
    
# this finds similarity, but I ended up not using it, and instead used a percentage to calculate similarity. 
# For example, if all the students matched perfectly on admit cohort, that varaible would have 100% similarity. If 7/10 groups matched perfectly, this was 70%
def get_similarity_stats(df, propensity_vars):
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
    
# calculates average treatment effect, lots of abstraction, as long as the dataset has the right variables, all you need to do is pass it in. 
# so, you might need to preprocess a little bit. 
def get_average_treatment_effect(df):
    
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
        
# calculates retention rates depending on a particular subgroup of interest
def calculate_retention_rates(df,test_name):
    print("calculating retentin for {}:".format(test_name))
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

# main retention method, very abstracted, just pass in the df
def get_retention_stats(df, path):

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
    
# helper
def calculate_mann_whitney_u(dist1, dist2, variable):
    
    u_stat, pval = mannwhitneyu(dist1[variable], dist2[variable])
    
    result = {
        'U': u_stat,
        'median1': dist1[variable].median(),
        'median2': dist2[variable].median(),
        'p-value': pval,
    }
    return result
    
def mann_whitney_to_csv():
    pass

# main mann_whitney_u method, abstracted: just pass in the dataset
def get_mann_whitney_u_stats(_df):
    
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
    

"""
def add_gpa_statistics(df):
    gpa = {
        'A+': 1,
        'A': 1,
        'A-': 0.925,
        'B+': 0.825,
        'B': 0.75,
        'B-': 0.675,
        'C+': 0.575,
        'C': 0.5,
        'C-': 0.425,
        'D+': 0.325,
        'D': 0.25,
        'D-': 0.175,
        'F+': 0,
        'F': 0,
        'F-': 0
    }
    units = {
        'CMPSC 40': 5,
        'CMPSC 196': 2
    }
    df_courses = pd.read_csv('All-data-OLD - ERSP_Courses.csv')
    df['GPA_Pre'] = df['F2GPA']
    df.loc[df['YR2 Match'] == 0, 'GPA_Pre'] = df.loc[df['YR2 Match'] == 0, 'F3GPA']
    df['GPA_Diff'] = df['GPA_Post'] - df['GPA_Pre']
    
    df['GPA_PRE'] = 0
    df['GPA_POST'] = 0
    
    for row_idx in range(df.shape[0]):
        current_row = df.iloc[row_idx]
        # print(current_row['ID'])
        current_grades = df_courses[df_courses['ID'] == current_row['ID']]
        
        
        pre_ersp_cutoff = 0
        if current_row['group'] == 'ERSP':
            pre_ersp_cutoff = current_row['ERSP_Cohort'] * 10 + 3
        else:
            if current_row['YR2 Match'] == 1:
                # 20184 => 20194 so 20193
                pre_ersp_cutoff = current_row['Admit_Cohort'] + 10 - 1
            else:
                # 20184 => 20204 so 20203
                pre_ersp_cutoff = current_row['Admit_Cohort'] + 20 - 1
                
        post_ersp_cutoff = pre_ersp_cutoff + 10
        
        pre_ersp_cutoff_year = pre_ersp_cutoff // 10
        pre_ersp_cutoff_qtr = pre_ersp_cutoff % 10
        post_ersp_cutoff_year = post_ersp_cutoff // 10
        post_ersp_cutoff_qtr = post_ersp_cutoff % 10
        
        # print("Admit: {}".format(current_row['Admit_Cohort']))
        # print("pre_ersp_cutoff: {}".format(pre_ersp_cutoff))
        # print("post_ersp_cutoff: {}".format(post_ersp_cutoff))
        grade_points_pre = 0
        grade_points_total_pre = 0
        grade_points_post = 0
        grade_points_total_post = 0
        
        for row_jdx in range(current_grades.shape[0]):
            course = current_grades.iloc[row_jdx]['Course']
            grade = current_grades.iloc[row_jdx]['Grade']
            quarter = current_grades.iloc[row_jdx]['Quarter'] % 10
            year = current_grades.iloc[row_jdx]['Quarter'] // 10
            # print("\t year: {} \n\tquarter: {}".format(year, quarter))
            if grade not in gpa.keys():
                # skip grade, does not count in current calculation
                continue
            
            total_points = 16
            if course in units.keys():
                total_points = units[course] * 4
                
            # decide whether to put it in pre or post calculations:
            if (year < pre_ersp_cutoff_year) or (year == pre_ersp_cutoff_year and quarter <= pre_ersp_cutoff_qtr):
                grade_points_total_pre += total_points
                grade_points_pre += gpa[grade] * total_points
            elif (year > post_ersp_cutoff_year) or (year == post_ersp_cutoff_year and quarter >= post_ersp_cutoff_qtr):
                grade_points_total_post += total_points
                grade_points_post += gpa[grade] * total_points
            
        if grade_points_total_pre != 0:
            df.at[row_idx,'GPA_PRE'] = (grade_points_pre / grade_points_total_pre) * 4
        if grade_points_total_post != 0:
            df.at[row_idx,'GPA_POST'] = (grade_points_post / grade_points_total_post) * 4
        
    # df[['ID', 'GPA_PRE', 'GPA_POST', 'Pre_ERSP_GPA']].to_csv('peek2.csv')
    # set cumulative post gpa
    post_gpa = pd.read_csv('ERSP_post_program_GPA - Sheet1.csv')
    for i in range(len(df.shape[0])):
        
        
        
    return df
"""

# the following gpa methods are not used, since the match file already provides the gpa information.
def add_gpa(df):
    df_gpa = pd.read_csv("ERSP_post_program_GPA - Sheet1.csv")
    
    df['cumgpa_3+'] = df_gpa['cumgpa_3+']
    df['cumgpa_4+'] = df_gpa['cumgpa_4+']
    
    return df
    
def add_gpa_statistics(df):
    
    df = add_gpa(df)
    
    df['GPA_Pre'] = df['F2GPA']
    df.loc[df['YR2 Match'] == 0, 'GPA_Pre'] = df.loc[df['YR2 Match'] == 0, 'F3GPA']
    
    df['GPA_Post'] = df['cumgpa_3+']
    df.loc[df['YR2 Match'] == 0, 'GPA_Post'] = df.loc[df['YR2 Match'] == 0, 'cumgpa_4+']
    
    df['GPA_Diff'] = df['GPA_Post'] - df['GPA_Pre']
    
    return df
    
    
def main():
    # the main function lays out how to use each method. 
    
    #pass in the file produced by the matching algorithm. 
    df = pd.read_csv('MATCHED_with_csceqtr copy.csv')
    
    # get a dataset with all just in case.
    df_all = pd.read_csv('All-data-NEW-ERSP_Full-Control-Group.csv')

    # convert to categorical just in case. 
    df_numerical, category_map = categorical_to_numerical(df)
    
    # variables of interest for the study (propensity is probably a bad name for this variable)
    propensity_vars = ['first_generation', 'Admit_Cohort', 'Major_at_Admission', 'CS_CE_qtr', 'eth_grp', 'SEX', 'GPA_Pre']
    
    print()
    print("DEMOGRAPHICS")
    get_demographic_stats(df_numerical, category_map)
    print("\tDemographic Stats reported in '" + save_prefix + "tables/Control_Demographics.csv' and '" + save_prefix + "tables/ERSP_Demographics.csv'\n")
    
    print("DIFFERENCE")
    get_difference_count(df_numerical, propensity_vars)
    print("\tGPA difference Stats reported in '" + save_prefix + "tables/gpa_difference_stats.csv'")
    print("\tVariable difference counts reported in '" + save_prefix + "tables/variable_difference_count.csv'")
    print("\tGPA difference plot reported in '" + save_prefix + "plots/GPA_Difference.png'\n")
    
    print("SIMILARITY")
    get_similarity_stats(df_numerical, propensity_vars)
    print("\tVariable similarity scores reported in '" + save_prefix + "tables/Similarity_Scores.csv'\n")
    
    print("ATE")
    print(df)
    get_average_treatment_effect(df)
    print("\tAverage Treatment Effect stats reported to '" + save_prefix + "tables/ate_stats.csv'\n")
    
    print("RETENTION")
    get_retention_stats(df, save_prefix + 'tables/Retention_Data.csv')
    get_retention_stats(df_all, save_prefix + 'tables/ALL_Retention_Data.csv')
    print("Retention Stats reported to 'Retention_Data.csv'\n")
    
    print("MANN WHITNEY U")
    get_mann_whitney_u_stats(df)
    print("Mann Whitney Stats reported to various tables and plots in '" + save_prefix + "plots' and '" + save_prefix + "tables'\n")
    
if __name__ == '__main__':
    main()