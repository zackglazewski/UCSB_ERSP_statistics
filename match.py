# IMPORTS

# Data Handling
import pandas as pd
import numpy as np

######################################################################################################################################################################

# FUNCTIONS - All methods have docstrings, please refer to them if you have any confusion. 

def match(matching_vars, df, df_orig):
    """Matches Control and Treated students based off of a filtering procedure based on the matching_vars argument.
    Variables that appear earlier in the list have higher priority. The priority order we used is used in main.
    The algorithm works like this:
    For every variable, get a filtered list of candidate matches. If there are no matches for that variable, skip it 
    and move on to the next one. Then, we select a final match from gpa using its nearest neighbor. 

    Args:
        matching_vars (List): A list of strings specifying variables to match on. Variables that come earlier
        have higher priority and will be matched on first. 
        df (DataFrame): A condensed form of the full dataset which only includes columns from the matching_vars list.
        df_orig (DataFrame): The original, uncondensed, dataset. Requires the columns: "F2GPA"  and "F3GPA" columns. 

    Returns:
       2-tuple (DataFrame, map) : Returns a DataFrame which includes info about what year they were matched on.
       The map is a mapping of ID's corresponding to matches.
    """
    print(df)
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

def write_matches_to_csv(df, matches, path):
    
    """Uses the ID mapping in the matches argument, to generate a dataset where each matched student
    appears in successive lines rather than dispersed among the data. 
    For example if ERSP student 1 matched with control student 8, and ERSP student 2 matched with control student 2, 
    the dataset would look like
    
    ERSP student 1
    
    Control student 8
    
    ERSP student 2
    
    Control student 2
    
    IMPORTANT: The file generated by this method is what stats.py will use to perform data analysis. 
    
    Args:
        df (DataFrame): the original dataset with control and treated students.
        matches (map): a mapping of matched IDs
        path (String): file to save the new dataset to. 
    """

    paired_df = pd.DataFrame()
    for key in matches.keys():
        paired_df = pd.concat([paired_df, df.loc[df['ID'] == key]])
        paired_df = pd.concat([paired_df, df.loc[df['ID'] == matches[key]]])
        
    paired_df.to_csv(path, index=False)

def add_gpa(df):
    """Adds general gpa information to the dataset. This has a hardcoded path from where I'm getting the new data from. 
    It is also hard-coded in terms of the names of columns. The data I was getting was called cumgpa_3+ and cumgpa4+. 
    These are cumulative gpas starting after each respective year. 
    
    For example cumgpa_3+ was the cumulative gpa starting in the 3rd year and continue onward. So, no information before the 3rd year
    is included in this cumulative number. 

    Args:
        df (DataFrame): The dataset to add the data to

    Returns:
        (DataFrame): The altered dataframe. 
    """
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

def add_gpa_statistics(df):
    """This adds essential gpa information for the stats.py stage of analysis. This function adds the Pre and Post GPA columns
    to the data AFTER you match. 

    Args:
        df (DataFrame): The dataset, which has already been through the matching algorithm. It should include the "YR2 Match" column. 

    Returns:
        (DataFrame): The same dataframe but with added gpa information. 
    """
    df = add_gpa(df)
    
    df['GPA_Pre'] = df['F2GPA']
    df.loc[df['YR2 Match'] == 0, 'GPA_Before'] = df.loc[df['YR2 Match'] == 0, 'F3GPA']
    
    df['GPA_Post'] = df['cumgpa_3+']
    df.loc[df['YR2 Match'] == 0, 'GPA_Post'] = df.loc[df['YR2 Match'] == 0, 'cumgpa_4+']
    
    df['GPA_Diff'] = df['GPA_Post'] - df['GPA_Pre']
    
    return df

######################################################################################################################################################################

# MAIN - This is the main function I used for my matching, depending on your data format, you may need to tweak hard-coded aspects in the analysis code. But, this should be
#        a good example of how to use each method, and how to set up their preconditions. 


def main():
    
    ######################################################################################################################################################################
    
    # STEP 1 - Read and Preprocess the data
    # STEP 1a: get the full control and ersp group together in one dataset. 
    df = pd.read_csv('All-data-NEW-ERSP_Full-Control-Group.csv')
    
    # STEP 1b: define the matching variables. Note: These are actually not propensity variables, but 
    #          I kept the name from when I was experimenting with propensity scores. But the name still
    #          works good to define variables that you want to match on I suppose. 
    propensity_variables = ['first_generation', 'Admit_Cohort', 'Major_at_Admission', 'CS_CE_qtr', 'eth_grp', 'SEX']
    
    # STEP 1c: Condense the original data set with the propensity variables
    df_condensed = df[['ID'] + propensity_variables + ['Pre_ERSP_GPA', 'F2GPA', 'F3GPA', 'ERSP_Cohort', 'group']].copy()
    
    ######################################################################################################################################################################

    # STEP 2 - Match!
    # STEP 2a: run the matching algorithm - see doc-strings of match for details
    df, id_map = match(propensity_variables, df_condensed, df)
    
    # STEP 2b: Add the Pre and Post gpa statistics to the newly matched dataset. See doc-strings for details
    df = add_gpa_statistics(df)

    # STEP 2c: Write the matches to a better format (to be used in stats.py)
    write_matches_to_csv(df, id_map, './MATCHED_test.csv')
    
    ######################################################################################################################################################################
    # Enjoy! If you have any questions feel free to contact me: email me at zglazewski@ucsb.edu

    return

if __name__ == '__main__':
    main()