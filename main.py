import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matching
import plots

saving_path = "C:/yoursavingpathhere"
files_path  = "C:/yourfilepathhere"


# Change based on your data type
raw_data = pd.read_excel(f"{files_path}/data.xlsx")

# Limit your dataset to pre-treatment time-period
pre_treatment = raw_data[raw_data['post_treatment']==0]

pre_treatment = pd.concat(
    [pre_treatment, pre_treatment.groupby('nsmoid')[["Time variant variables here"]
         .transform('mean')
         .rename(columns=lambda x: f"avg_{x}")],
    axis=1
)

# Only keep one pbservation per ID
pre_treatment = pre_treatment.drop_duplicates(subset=['ID'],keep='last')

# List of variables for the logit regression. 
treatment_col    = 'yourtreatment'
categorical_vars = ['yourcategoricalvariables'] 
dummies          = pd.get_dummies(pre_treatment[['ID'] + categorical_vars], columns=categorical_vars, drop_first=True, dtype=int)
# including all dummy variables, exclusing the generate list above
noncategorical   = ['yourvariables'] 

matching_data = pre_treatment[[treatment_col] + ['ID'] + noncategorical].replace([np.inf, -np.inf], np.nan).dropna()
matching_data = pd.merge(matching_data, dummies, on='ID')
covariates = noncategorical + dummies.columns.tolist()

matching_data['pscore'] = sm.Logit(matching_data[treatment_col], sm.add_constant(matching_data[covariates])).fit(disp=0).predict()


# Get the 10th smallest and 10th largest values (Lechner’s Approach)
lower_bound = max(
    matching_data[matching_data[treatment_col] == 1]['pscore'].nsmallest(10).iloc[-1],
    matching_data[matching_data[treatment_col] == 0]['pscore'].nsmallest(10).iloc[-1]
)
upper_bound = min(
    matching_data[matching_data[treatment_col] == 1]['pscore'].nlargest(10).iloc[-1],
    matching_data[matching_data[treatment_col] == 0]['pscore'].nlargest(10).iloc[-1]
)
matching_data = matching_data[(matching_data['pscore'] <= upper_bound)&(matching_data['pscore'] >= lower_bound)]


nn_matched= matching.NN(data=matching_data, numberofneighbors=number_match, treatment=treatment_col)
nn_matched  = raw_data[raw_data['ID'].isin(nn_matched['ID'].unique())]
nn_matched = nn_matched.merge(pre_treatment[['ID', 'pscore']], on='ID', how='left')

plot_name = 'addplotname'
plots.mirrored_histogram(data=nn_matched, treatment=treatment_col, score='pscore', bins=40,name=plot_name)

output_filename = 'matcheddataname'
matched_data.to_excel(f"{files_path}/{output_filename}.xlsx", index=False)

