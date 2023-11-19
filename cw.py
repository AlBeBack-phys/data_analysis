import pandas as pd
import numpy as np
from scipy.stats import t, ttest_ind
import json
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare, norm
import matplotlib.pyplot as plt


df = pd.read_csv('mussl.csv')
selected_data = df[(df['Month'] == 'May') & df['Site'].isin(['S4', 'S5', 'S6'])]
results = []
for site1 in ['S4', 'S5', 'S6']:
    for site2 in ['S4', 'S5', 'S6']:
        if site1 < site2:
            site1_length_data = selected_data[selected_data['Site'] == site1]['Length'].apply(np.log)
            site2_length_data = selected_data[selected_data['Site'] == site2]['Length'].apply(np.log)
            t_statistic, p_value = ttest_ind(site1_length_data, site2_length_data)
            degrees_of_freedom = len(site1_length_data) + len(site2_length_data) - 2
            significance_level = 0.05
            t_critical = t.ppf(1 - significance_level / 2, degrees_of_freedom)
            if p_value >= significance_level:
                results.append({'left': site1,
                'right': site2,
                'pvalue': p_value,
                'statistic': t_statistic,
                'threshold': t_critical})


with open('ttest.json', 'w') as file:
    json.dump(results, file, indent=2)


site5_data = df[df['Site'] == 'S5']
X = site5_data['Day'].values.reshape(-1, 1)
y = np.log(site5_data['Length'])
regression_model = LinearRegression().fit(X, y)
residuals = y - regression_model.predict(X)
unit_error_variance = np.mean(residuals ** 2)
standardized_residuals = residuals / np.sqrt(unit_error_variance)
hist, bins = np.histogram(standardized_residuals, bins=6)
frequency = bins.tolist()
chi2_pvalue, chi2_statistic = chisquare(hist)
with open('chi2.json', 'w') as file:
    json.dump({'freq': frequency, 'pvalue': chi2_pvalue, 'statistic': chi2_statistic}, file, indent=2)

expected = norm.pdf((bins[:-1]+bins[1:])/2*len(standardized_residuals))
plt.hist(standardized_residuals, bins=frequency, density=True, alpha=0.5, label='Emirical')
plt.plot((bins[:-1] + bins[1:])/2, expected, 'r-', label='Theoretical')
plt.legend()
plt.savefig('chi2.png')
