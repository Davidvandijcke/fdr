import pandas as pd 
import os
import numpy as np

# get relative dir
dir = os.path.dirname(__file__)

# get directory above
main_dir = os.path.dirname(os.path.dirname(dir))
data_in = os.path.join(main_dir, 'data', 'in')

# read orbis
orbis = pd.read_csv(os.path.join(data_in, 'orbis', 'query.csv'))





# read PPP data
ppp = pd.read_csv(os.path.join(data_in, 'ppp', 'public_up_to_150k_2_230331.csv'))


ppp = pd.read_csv(os.path.join(data_in, 'ppp', 'public_150k_plus_230331.csv'))
ppp = ppp[ppp['ProcessingMethod'] == 'PPP']


ppp['total_proceed'] = (ppp['UTILITIES_PROCEED'].fillna(0) +
                        ppp['PAYROLL_PROCEED'].fillna(0) +
                        ppp['MORTGAGE_INTEREST_PROCEED'].fillna(0) +
                        ppp['RENT_PROCEED'].fillna(0) +
                        ppp['REFINANCE_EIDL_PROCEED'].fillna(0) +
                        ppp['HEALTH_CARE_PROCEED'].fillna(0) +
                        ppp['DEBT_INTEREST_PROCEED'].fillna(0))


(ppp['total_proceed'] == ppp['InitialApprovalAmount']).mean()

ppp['payroll_percent'] = (ppp['PAYROLL_PROCEED'] ) /  ppp['CurrentApprovalAmount']
# fill infinity values with 0   
ppp.loc[ppp['payroll_percent'] == float('inf'), 'payroll_percent'] = np.nan

ppp.loc[ppp['ForgivenessAmount'].isna(), 'ForgivenessAmount'] = 0
ppp['forgiven'] = (ppp['ForgivenessAmount'] > 0).astype(int)
ppp['ForgivenessPercent'] = ppp['ForgivenessAmount'] / ppp['total_proceed']

# aggregate forgiven into bins along the payroll_percent variable and plot
# group data and get average 'forgiven'
df = ppp.groupby(pd.cut(ppp['payroll_percent'], bins=100)).agg({'InitialApprovalAmount': 'mean'}).reset_index()

# convert intervals to their mid points
df['payroll_percent'] = (df['payroll_percent'].apply(lambda x: x.mid)).astype(float)

# plot data
df.plot(x='payroll_percent', y='InitialApprovalAmount')

ppp[ppp.payroll_percent < 1].plot(x = 'payroll_percent', y = 'InitialApprovalAmount', kind = 'scatter')



# plot vertical line at 0.6
plt.axvline(x=0.61, color='red')


import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(data=ppp, x='payroll_percent', y='forgiven')




# count unique borrowers
ppp['BorrowerName'].unique().shape[0]