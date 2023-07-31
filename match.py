import pandas as pd
import numpy as np
import math
import sys

# decides where the results will be stored
save_prefix = './data_w_csceqtr/'

# the main matching algorithm, it basically filters the data
def match(matching_vars, df, df_orig):
    print(df)
    """Matches Two IDs from df based off of values in matching_vars.
    If a match is not found, the variable is skipped

    Returns:
        (DataFrame, Dict): DataFrame: A df with matching entry filled, Dict: A mapping of IDs
    """
    df_o = df_orig.copy()
    df_o['YR2 Match'] = 0
    # Separate Treated and Control Group
    treated_df = df.loc[df['group'] == 'ERSP']
    control_df = df.loc[df['group'] == 'Control']
    
    matching = {}
    
    # Necessary for index iteration
    treated_df = treated_df.reset_index(drop=True)
    
    # For every ERSP Student
    for idx in range(treated_df.shape[0]):
        filtered = control_df.copy()
        # Filter for each variable
        for var in matching_vars:
            
            target_match = treated_df.at[idx,var]
            
            prev = filtered
            filtered = filtered[filtered[var] == target_match]
            if filtered.empty:
                # revert back before value was filtered / skip this var
                filtered = prev
                continue

        year = (treated_df.at[idx, 'ERSP_Cohort'] - treated_df.at[idx, 'Admit_Cohort'] // 10) + 1
        gpa_var = 'F2GPA'
        if (year == 3):
            gpa_var = 'F3GPA'
        # gpa_var = 'GPA_PRE_yr2'
        # if (year == 3):
        #     gpa_var = 'GPA_PRE_yr3'
            
      
        # After filtering all the variables, get closest neighbor for GPA variable
        filtered_copy = filtered.copy()
        filtered_copy['GPA_Difference'] = abs(filtered_copy[gpa_var] - treated_df.at[idx,'Pre_ERSP_GPA'])
        
        matching[treated_df.at[idx,'ID']] = filtered_copy[filtered_copy['GPA_Difference'] == filtered_copy['GPA_Difference'].min()].iloc[0]['ID']
        # treated_df.loc[idx,'MATCH'] = matching[treated_df.loc[idx,'ID']]
        
        # Drop the matching control student from the control_df so it cannot be matched again (without replacement)
        control_df = control_df[control_df['ID'] != matching[treated_df.at[idx,'ID']]]
        if gpa_var == 'F2GPA':
            # df_o[df_o.ID == matching[treated_df.at[idx,'ID']]]['YR2 Match'] = 1
            # df_o[df_o.ID == treated_df.at[idx,'ID']]['YR2 Match'] = 1
            
            df_o.loc[df.ID == matching[treated_df.at[idx,'ID']], 'YR2 Match'] = 1
            df_o.loc[df.ID == treated_df.at[idx,'ID'], 'YR2 Match'] = 1
        else:
            # df_o[df_o.ID == matching[treated_df.at[idx,'ID']]]['YR2 Match'] = 0
            # df_o[df_o.ID == treated_df.at[idx,'ID']]['YR2 Match'] = 0
            df_o.loc[df.ID == matching[treated_df.at[idx,'ID']], 'YR2 Match'] = 0
            df_o.loc[df.ID == treated_df.at[idx,'ID'], 'YR2 Match'] = 0

    return df_o, matching

# finds which quarter is the first after ersp
def generate_post_quarter_column(_df):
    
    df = _df[['ID', 'Admit_Cohort', 'ERSP_Cohort', 'group']].copy()
    year_of_admission = df['Admit_Cohort'].apply(lambda x: x // 10).copy()
    
    add1 = lambda x: int((x+1) * 10 + 3)
    add2 = lambda x: int((x+2) * 10 + 3)
    add3 = lambda x: int((x+3) * 10 + 3)
    def set_neg(x):
        return -1
    
    df['Quarter_after_ERSP'] = df[df.group == 'ERSP']['ERSP_Cohort'].apply(add1).copy()
    
    df['Quarter_3rd_year'] = year_of_admission.apply(add2).copy()
    df['Quarter_4th_year'] = year_of_admission.apply(add3).copy()
    
    df.to_csv('post_quarter_data.csv', float_format='%.0f', index=False)
    
# for a secondary task, if you wanted to split the data based on ersp cohort
def split_by_cohort(df):
    
    """Splits ERSP students based on their year when participating in ERSP
    """
    print(df)
    df = df.loc[df['group'] == 'ERSP']
    year_of_admission = df['Admit_Cohort'].apply(lambda x: x // 10).copy()

    df['Year_Of_ERSP'] = df['ERSP_Cohort'] - year_of_admission + 1
    add1 = lambda x: int((x+1) * 10 + 3)
    df['Quarter_after_ERSP'] = df['ERSP_Cohort'].apply(add1)
    print(df)
    
    group1 = df.loc[df['Year_Of_ERSP'] == 2]
    group2 = df.loc[df['Year_Of_ERSP'] == 3]
    
    print(group1)
    print(group2)
    
    df.to_csv('ERSP_with_class_years.csv')
    group1.to_csv('second_year_group.csv')
    group2.to_csv('third_year_group.csv')

# specifically pairs matched students in successive lines rather than dispersed among the data
def write_matches_to_csv(df, matches, path):
    
    """Takes in Dict of ID matches, df, path to csv, and a categorical mapping and 
    writes only the matches to a csv.
    """

    paired_df = pd.DataFrame()
    for key in matches.keys():
        paired_df = pd.concat([paired_df, df.loc[df['ID'] == key]])
        paired_df = pd.concat([paired_df, df.loc[df['ID'] == matches[key]]])
        
    paired_df.to_csv(path, index=False)
    
# converts categorical data to numerical by enumeration, gives an id mapping so the real category isn't forgotten
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

# if I wanted to have command line arguments, but unused in the end because not really needed
def print_usage():
    print("Incorrect Command-line Arguments: ")
    print("python3 match.py [input filepath] [output filepath] [propensity variables in order of priority]")

# counts differences between matched variables.
def count_differences(df, matches, matching_vars):
    
    """Counts differences for each column between each match

    Returns:
        Dict: Key values pairs as follows, col_name: differences
    """
    
    print(df)
    differences = {}
    for var in matching_vars:
        differences[var] = 0
    
    for key in matches.keys():
        temp = pd.DataFrame()
        temp = pd.concat([temp, df.loc[df['ID'] == key]])
        temp = pd.concat([temp, df.loc[df['ID'] == matches[key]]])
        temp.drop('ID', axis=1, inplace=True)
        diffs = temp.diff().copy()
        diffs = diffs.reset_index(drop=True)
        for var in matching_vars:
            if diffs.loc[1,var] != 0:
                differences[var] += 1
                
    return differences
    
# manual gpa calculations, but unused after new data was given
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
    # df['GPA_Before'] = df['F2GPA']
    # df.loc[df['YR2 Match'] == 0, 'GPA_Before'] = df.loc[df['YR2 Match'] == 0, 'F3GPA']
    # df['GPA_DIFF'] = df['Last_Quarter_GPA'] - df['GPA_Before']
    
    df['GPA_PRE_yr2'] = 0
    df['GPA_PRE_yr3'] = 0
    
    for row_idx in range(df.shape[0]):
        current_row = df.iloc[row_idx]
        # print(current_row['ID'])
        current_grades = df_courses[df_courses['ID'] == current_row['ID']]
        
        
        pre_ersp_cutoff_yr2 = 0
        pre_ersp_cutoff_yr3 = 0
        
        pre_ersp_cutoff_yr2 = current_row['Admit_Cohort'] + 10 - 1

        pre_ersp_cutoff_yr3 = current_row['Admit_Cohort'] + 20 - 1

        
        pre_ersp_cutoff_year_yr2 = pre_ersp_cutoff_yr2 // 10
        pre_ersp_cutoff_qtr_yr2 = pre_ersp_cutoff_yr2 % 10
        pre_ersp_cutoff_year_yr3 = pre_ersp_cutoff_yr3 // 10
        pre_ersp_cutoff_qtr_yr3 = pre_ersp_cutoff_yr3 % 10
        
        # print("Admit: {}".format(current_row['Admit_Cohort']))
        # print("pre_ersp_cutoff: {}".format(pre_ersp_cutoff))
        # print("post_ersp_cutoff: {}".format(post_ersp_cutoff))
        grade_points_pre_yr2 = 0
        grade_points_total_pre_yr2 = 0
        grade_points_pre_yr3 = 0
        grade_points_total_pre_yr3 = 0
        
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
            if (year < pre_ersp_cutoff_year_yr2) or (year == pre_ersp_cutoff_year_yr2 and quarter <= pre_ersp_cutoff_qtr_yr2):
                grade_points_total_pre_yr2 += total_points
                grade_points_pre_yr2 += gpa[grade] * total_points
            if (year < pre_ersp_cutoff_year_yr3) or (year == pre_ersp_cutoff_year_yr3 and quarter >= pre_ersp_cutoff_qtr_yr3):
                grade_points_total_pre_yr3 += total_points
                grade_points_pre_yr3 += gpa[grade] * total_points
            
        if grade_points_total_pre_yr2 != 0:
            df.at[row_idx,'GPA_PRE_yr2'] = (grade_points_pre_yr2 / grade_points_total_pre_yr2) * 4
        if grade_points_total_pre_yr3 != 0:
            df.at[row_idx,'GPA_PRE_yr3'] = (grade_points_pre_yr3 / grade_points_total_pre_yr3) * 4
        
    df[['ID', 'GPA_PRE_yr2', 'GPA_PRE_yr3', 'Pre_ERSP_GPA']].to_csv('peek.csv')
    return df
    
"""
# adds general gpa information to the dataset
def add_gpa(df):
    df_gpa = pd.read_csv("ERSP_post_program_GPA - Sheet1.csv")
    print(df_gpa.columns)
    df['cumgpa_3+'] = df_gpa['cumgpa_3+']
    df['cumgpa_4+'] = df_gpa['cumgpa4+']
    # print(df[['cumgpa_3+', 'cumgpa_4+']])
    def replace_non_numeric(x):
        if pd.isna(x):
            return 0
        try:
            x = float(x)
            if pd.isna(x):
                return 0
            else:
                return x
        except ValueError:
            return 0

    df['cumgpa_3+'] = df['cumgpa_3+'].apply(replace_non_numeric)
    df['cumgpa_4+'] = df['cumgpa_4+'].apply(replace_non_numeric)
    print(df[['cumgpa_3+', 'cumgpa_4+']])
    
    return df

# adds gpa information after matching
def add_gpa_statistics(df):
    df = add_gpa(df)
    
    df['GPA_Pre'] = df['F2GPA']
    df.loc[df['YR2 Match'] == 0, 'GPA_Before'] = df.loc[df['YR2 Match'] == 0, 'F3GPA']
    
    df['GPA_Post'] = df['cumgpa_3+']
    df.loc[df['YR2 Match'] == 0, 'GPA_Post'] = df.loc[df['YR2 Match'] == 0, 'cumgpa_4+']
    
    df['GPA_Diff'] = df['GPA_Post'] - df['GPA_Pre']
    
    return df

# generates a table of demographic percentages
def get_demographic_stats(df, category_map):
    treated_stats = np.zeros(shape=(len(category_map['SEX']), len(category_map['eth_grp'])))
    control_stats = np.zeros(shape=(len(category_map['SEX']), len(category_map['eth_grp'])))
    print(category_map)
    for sex_id in category_map['SEX'].keys():
        for eth_id in category_map['eth_grp'].keys():
            treated_stats[sex_id,eth_id] = len(df[(df.SEX == sex_id) & (df.eth_grp == eth_id) & (df.group == 0)])
            control_stats[sex_id,eth_id] = len(df[(df.SEX == sex_id) & (df.eth_grp == eth_id) & (df.group == 1)])
            
   
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
            
    
def main():
    
    df = pd.read_csv('All-data-NEW-ERSP_Full-Control-Group.csv')
    
    # not actually using propensity scores, just the name I gave to the variables of interest.
    propensity_variables = ['first_generation', 'Admit_Cohort', 'Major_at_Admission', 'CS_CE_qtr', 'eth_grp', 'SEX']
    
    # get data with just numerical data
    num_df, mapping = categorical_to_numerical(df)
    get_demographic_stats(num_df, mapping)
    
    
    # dataset with only relevant variables
    df_condensed = df[['ID'] + propensity_variables + ['Pre_ERSP_GPA', 'F2GPA', 'F3GPA', 'ERSP_Cohort', 'group']].copy()
    
    # matching algorithm
    df, id_map = match(propensity_variables, df_condensed, df)
    
    # need to add specific gpa data based on if a control student was matched with a 3rd year or 4th year student
    df = add_gpa_statistics(df)

    # write to a file
    write_matches_to_csv(df, id_map, './MATCHED_test.csv')

    return

if __name__ == '__main__':
    main()