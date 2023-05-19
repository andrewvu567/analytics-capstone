# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 01:47:25 2023

@author: avaz5
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Read data.
df = pd.read_csv('Base.csv')

# Segment for accounts for use with histogram plotting + t tests
fraud_accounts = df[df['fraud_bool'] == 1]
legit_accounts = df[df['fraud_bool'] == 0] 

#%%
# Make cumulative distribution function charts + bar charts of averages for numeric variables
# segmented by fraud/not fraud.

numeric_cols = ['name_email_similarity','prev_address_months_count',
                'current_address_months_count','days_since_request',
                'intended_balcon_amount','zip_count_4w','velocity_6h','velocity_24h',
                'velocity_4w','bank_branch_count_8w','date_of_birth_distinct_emails_4w',
                'credit_risk_score','bank_months_count','proposed_credit_limit',
                'session_length_in_minutes','device_distinct_emails_8w', 'device_fraud_count']

cols_with_imputes = ['bank_months_count', 'prev_address_months_count', 'current_address_months_count']


# CDFs
for col in numeric_cols:
    # These columns have imputed values for missing values, do not want to plot them.
    if col not in cols_with_imputes:
        cdf = sns.ecdfplot(data = df, x = col, hue = 'fraud_bool')
    else:
        # Create a temporary data frame where records where there was an imputed -1 for that variable are excluded
        no_imputes_df = df[df[col] != -1]
        cdf = sns.ecdfplot(data = no_imputes_df, x = col, hue = 'fraud_bool')
    
    cdf_fig = cdf.get_figure()
    cdf_fig.savefig(f'./visualizations/cdf/{col}_cdf.png')
    plt.show()
    

# Bar charts
for col in numeric_cols:
    fig, ax = plt.subplots()
    if col not in cols_with_imputes:
        bar = ax.bar(x = df.groupby('fraud_bool')[col].mean().index.astype('str'), 
                     height = df.groupby('fraud_bool')[col].mean())
    else:
        no_imputes_df = df[df[col] != -1]
        bar = ax.bar(x = no_imputes_df.groupby('fraud_bool')[col].mean().index.astype('str'), 
               height = no_imputes_df.groupby('fraud_bool')[col].mean())
        
    ax.set_title(f'Mean {col} by fraud_bool')
    ax.bar_label(bar, label_type = 'center', labels = [f'{x:,.2f}' for x in bar.datavalues],
                 color = '#dedcd5')
    plt.tight_layout()
    plt.savefig(f'./visualizations/numerical_bars/{col}.png')
    plt.show()
    
#%%
# Make bar charts for categorical variables
categorical_cols = ['payment_type','employment_status','email_is_free','housing_status',
                    'phone_home_valid','phone_mobile_valid','has_other_cards',
                    'foreign_request','source','device_os','keep_alive_session', 'month',
                    'customer_age', 'income']

for col in categorical_cols:
    plot_data_df = pd.DataFrame({f'{col}': df.groupby([f'{col}'])['fraud_bool'].mean().index.astype('str'), 
                                 'avg_fraud_rate' :df.groupby([f'{col}'])['fraud_bool'].mean()})
    if col == 'income':
        # need to handle conversion of existing floating point values for income into strings. 
        # Conversion added extra 0s to the end of certain values
        plot_data_df['income'] = plot_data_df['income'].apply(lambda x: x[:3] if len(x) > 3 else x[:3])
    
    plot_data_df.sort_values(by = 'avg_fraud_rate', ascending = False, inplace = True)
    
    fig, ax = plt.subplots()
    
    bar2 = ax.bar(x = plot_data_df[f'{col}'], height = plot_data_df['avg_fraud_rate'])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 1))
    ax.set_title(f'Mean fraud rate by each value of {col}')
    ax.bar_label(bar2, label_type = 'center', labels = [f'{x:.2%}' for x in bar2.datavalues],
                 color = '#dedcd5')
    plt.tight_layout()
    
    plt.savefig(f'./visualizations/barcharts/{col}.png')
    plt.show()

#%%
# Perform a proportion difference test between the two groups for categorical variables where there's
# 2+ groups
# allows import of r package
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
stats = importr('stats')

for col in categorical_cols:
    prop = stats.prop_test(x = df.groupby([col])['fraud_bool'].sum().values,
                           n = df.groupby([col])['fraud_bool'].count().values)
    print(f'Proprtions test results for {col}\n{prop}')

#%%
# t test (numerical values)
import statistics
from scipy import stats


# test equivalent variances (condition)
for col in numeric_cols:
    if col not in cols_with_imputes:
        hi = max(statistics.variance(fraud_accounts[col].values), statistics.variance(legit_accounts[col].values))
        lo = min(statistics.variance(fraud_accounts[col].values), statistics.variance(legit_accounts[col].values))
    else:
        fraud_imputes_removed = fraud_accounts[fraud_accounts[col] != -1][col]
        legit_imputes_removed = legit_accounts[legit_accounts[col] != -1][col]
        hi = max(statistics.variance(fraud_imputes_removed.values), statistics.variance(legit_imputes_removed.values))
        lo = min(statistics.variance(fraud_imputes_removed.values), statistics.variance(legit_imputes_removed.values))
    
    print(f'Variance ratio for {col} is {hi/lo}')
        
        
# histogram plots to test normality
for col in numeric_cols:
    fig, axes = plt.subplots(2,1)
    if col not in cols_with_imputes:
        axes[0].hist(fraud_accounts[f'{col}'], alpha = 0.50)
        axes[0].set_title(f'{col} distribution for fraudulent accounts')
        axes[1].hist(legit_accounts[f'{col}'], alpha = 0.50)
        axes[1].set_title(f'{col} distribution for legitimate accounts')
    else:
        fraud_imputes_removed = fraud_accounts[fraud_accounts[col] != -1]
        legit_imputes_removed = legit_accounts[legit_accounts[col] != -1]
        axes[0].hist(fraud_imputes_removed[f'{col}'], alpha = 0.50)
        axes[0].set_title(f'{col} distribution for fraudulent accounts')
        axes[1].hist(legit_imputes_removed[f'{col}'], alpha = 0.50)
        axes[1].set_title(f'{col} distribution for legitimate accounts')
        
    plt.tight_layout()
    fig.savefig(f'./visualizations/histograms/{col}.png')

# t-test
for col in numeric_cols:
    if col not in ['prev_address_months_count', 'current_address_months_count', 'bank_months_count']:
        print(f'T-Test results for {col}: ' + str(stats.ttest_ind(a = fraud_accounts[col], 
                                                                  b = legit_accounts[col])))
    else:
        fraud_imputes_removed = fraud_accounts[fraud_accounts[col] != -1]
        legit_imputes_removed = legit_accounts[legit_accounts[col] != -1]
        print(f'T-Test results for {col}: ' + str(stats.ttest_ind(a = fraud_imputes_removed[col].values,
                                                                  b = legit_imputes_removed[col].values)))

        

