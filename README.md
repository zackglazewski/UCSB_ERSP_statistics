# Getting Started
There are some sections of the code (in both files) which are hard-coded to suit the structure of the data I was using. 
If you find your data having slightly different columns / column names, I suggest doing a find all search for "hard-coded" or "hard coded", as I made an effort to spot them all. 

DISCLAIMER: If you are heavily relying on this code to get your results, I HIGHLY recommend reading and going through the doc-strings I wrote in each function. I wrote all these functions for my own reference and use without the knowledge that I'd be passing them down to someone else to use, so some parts might be weird or confusing. I've tried my best to document these functions for you. 

# Matching
See the main method in match.py for details on how to match. (highly recommended). The same steps provided below are also included in the main method. 

You will be following these steps:
#### STEP 1 - Read and Preprocess the data
###### STEP 1a: 
Get the full control and ersp group together in one dataset. 
###### STEP 1b: 
Define the matching variables. Note: These are actually not propensity variables, but I kept the name from when I was experimenting with propensity scores. But the name still works good to define variables that you want to match on I suppose.
###### STEP 1c: 
Condense the original data set with the propensity variables

#### STEP 2 - Match! 
###### STEP 2a: 
run the matching algorithm - see doc-strings of match for details
###### STEP 2b: 
Add the Pre and Post gpa statistics to the newly matched dataset. See doc-strings for details
###### STEP 2c: 
Write the matches to a better format (to be used in stats.py)


IMPORTANT! The resulting file of match.py will be used as the input file to stats.py

# Analysis
See the main method in stats.py for details on how to perform the analysis (highly recommended)

You will be following these steps:
#### STEP 1 - Prep the data
###### STEP 1a: 
Start out with the file produced by match.py. For me, this was called "MATCHED_with_csceqtr copy.csv". Pay attention to the name of the file you chose.
###### STEP 1b: 
Get all the data just in case. I happened to use it to generate retention data for the entire control group and compare to the matched control group. 
###### STEP 1c: 
Get an all-numerical dataset, as some methods require it.
###### STEP 1d: 
Select the variables of interest for similarity and difference checks. I used this list for getting difference counts.

#### STEP 2 - Get Stats for the Matching algorithm
###### STEP 2a: 
Pass in numerical data and the coresponding map - see details in get_demographic_stats's docstring
###### STEP 2b: 
Find the number of differences between variables of interest. Pass in a numerical dataset and variables of interest. See the doc-string of get_difference_count for details.
###### STEP 2c: 
Find the similarity measure of the matched group: WARNING: I DID NOT USE THIS METHOD, see get_similarity_stats's doc-string for details. 

#### STEP 3 -  Main Objective Analysis - ATE, Retention, and Mann Whitney U.
###### STEP 3a: 
Find the average treatment effect. Pass in the matched dataset. See the doc-string for get_average_treatment_effect for details.
###### STEP 3b: 
Find the retention among the matched groups. Here, I am doing analysis between both the matched only group and the entire control group. See the doc-string for get_retention_stats for details.
###### STEP 3c: 
Find the Mann Whitney U stats. Pass in the entire matched dataset. See the doc-string of get_mann_whitney_u_stats for details. 

# Contact Info
Enjoy! If you have any questions, feel free to contact me at zglazewski@ucsb.edu