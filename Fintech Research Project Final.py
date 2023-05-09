import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.iv import IVGMM
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('ETF_prices.csv')

# Convert 'price_date' to datetime format
data['price_date'] = pd.to_datetime(data['price_date'])

# Extract the year from 'price_date' and create a new 'year' column
data['year'] = data['price_date'].dt.year

# Calculate the daily return
data['daily_return'] = data.groupby('fund_symbol')['adj_close'].pct_change()

# Calculate the average yearly volume
data['avg_yearly_volume'] = data.groupby(['fund_symbol', 'year'])['volume'].transform(np.mean)

# Calculate the log of average yearly volume
data['avg_yearly_log_volume'] = np.log(data['avg_yearly_volume'])

# Calculate the yearly return
data['yearly_return'] = data.groupby(['fund_symbol', 'year'])['daily_return'].transform(np.sum)

# Drop duplicates
data = data.drop_duplicates(subset=['fund_symbol', 'year'])

# Filter out ETFs with less than 2 years of data
data = data.groupby('fund_symbol').filter(lambda x: len(x) >= 2)

# Create a MultiIndex with 'fund_symbol' and 'year' as the indices
data = data.set_index(['fund_symbol', 'year'])

# Generate lagged variables
data['lag_yearly_return'] = data.groupby('fund_symbol')['yearly_return'].shift(1)
data['lag_avg_yearly_log_volume'] = data.groupby('fund_symbol')['avg_yearly_log_volume'].shift(1)

# Drop rows with NaN values for the lagged variables
data = data.dropna(subset=['lag_yearly_return', 'lag_avg_yearly_log_volume'])

# Drop rows with NaN or infinite values in 'yearly_return', 'avg_yearly_log_volume', 'lag_yearly_return', and 'lag_avg_yearly_log_volume'
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=['yearly_return', 'avg_yearly_log_volume', 'lag_yearly_return', 'lag_avg_yearly_log_volume'])

# Define the models
simple_model = smf.ols('yearly_return ~ avg_yearly_log_volume', data=data).fit()
pooled_ols_model = PanelOLS.from_formula('yearly_return ~ avg_yearly_log_volume', data=data).fit()
random_effects_model = RandomEffects.from_formula('yearly_return ~ avg_yearly_log_volume', data=data).fit()
fixed_effects_model = PanelOLS.from_formula('yearly_return ~ avg_yearly_log_volume + lag_yearly_return + lag_avg_yearly_log_volume + EntityEffects', data=data).fit()
gmm_model = IVGMM.from_formula('yearly_return ~ 1 + avg_yearly_log_volume + [lag_yearly_return ~ lag_avg_yearly_log_volume]', data=data).fit()

# Scatter plot for the Simple Regression Model
data_for_plot = data.reset_index()
plt.figure(figsize=(10, 6))
sns.regplot(data=data_for_plot, x='avg_yearly_log_volume', y='yearly_return', ci=None, scatter_kws={'alpha': 0.5})
plt.title('Simple Regression Model: Yearly Return vs. Average Yearly Log Volume')
plt.xlabel('Average Yearly Log Volume')
plt.ylabel('Yearly Return')
plt.show()

# Scatter plot for the Fixed-Effects Panel Regression Model (without EntityEffects)
plt.figure(figsize=(10, 6))
sns.regplot(data=data_for_plot, x='avg_yearly_log_volume', y='yearly_return', ci=None, scatter_kws={'alpha': 0.5})
plt.title('Fixed-Effects Panel Regression Model (without EntityEffects): Yearly Return vs. Average Yearly Log Volume')
plt.xlabel('Average Yearly Log Volume')
plt.ylabel('Yearly Return')
plt.show()

# Scatter plot for the Random-Effects Panel Regression Model
plt.figure(figsize=(10, 6))
sns.regplot(data=data_for_plot, x='avg_yearly_log_volume', y='yearly_return', ci=None, scatter_kws={'alpha': 0.5})
plt.title('Random-Effects Panel Regression Model: Yearly Return vs. Average Yearly Log Volume')
plt.xlabel('Average Yearly Log Volume')
plt.ylabel('Yearly Return')
plt.show()

# Scatter plot for the Pooled OLS Regression Model
plt.figure(figsize=(10, 6))
sns.regplot(data=data_for_plot, x='avg_yearly_log_volume', y='yearly_return', ci=None, scatter_kws={'alpha': 0.5})
plt.title('Pooled OLS Regression Model: Yearly Return vs. Average Yearly Log Volume')
plt.xlabel('Average Yearly Log Volume')
plt.ylabel('Yearly Return')
plt.show()

# Scatter plot for the IV-GMM Regression Model
plt.figure(figsize=(10, 6))
sns.regplot(data=data_for_plot, x='avg_yearly_log_volume', y='yearly_return', ci=None, scatter_kws={'alpha': 0.5})
plt.title('IV-GMM Regression Model: Yearly Return vs. Average Yearly Log Volume')
plt.xlabel('Average Yearly Log Volume')
plt.ylabel('Yearly Return')
plt.show()

import statsmodels.iolib.summary2 as summary2

# Print the summary for Simple Regression Model
print("Simple Regression Model:")
print(simple_model.summary())

# Print the summary for Pooled OLS Model
print("\nPooled OLS Model:")
print(pooled_ols_model.summary)

# Print summaries for Random-Effects, Fixed-Effects, and IV-GMM models
print("Random-Effects Panel Regression Model:")
print(random_effects_model.summary)

print("Fixed-Effects Panel Regression Model:")
print(fixed_effects_model.summary)

print("IV-GMM Regression Model:")
print(gmm_model.summary)

# Extract key measures from each model
models = {
    'Simple Regression': simple_model,
    'Pooled OLS': pooled_ols_model,
    'Random Effects': random_effects_model,
    'Fixed Effects': fixed_effects_model,
    'IV-GMM': gmm_model
}

model_measures = []

for name, model in models.items():
    model_measures.append({
        'Model': name,
        'R-squared': model.rsquared,
        'Adj. R-squared': model.rsquared_adj if hasattr(model, 'rsquared_adj') else None,
        'F-statistic': model.fvalue if hasattr(model, 'fvalue') else model.f_statistic.stat,
        'p-value': model.f_pvalue if hasattr(model, 'f_pvalue') else None,
        'coef_avg_yearly_volume': model.params['avg_yearly_log_volume'] if 'avg_yearly_log_volume' in model.params.index else model.params[1]
    })

    # Create a DataFrame to display the key measures
comparison_table = pd.DataFrame(model_measures)
comparison_table = comparison_table.set_index('Model')
comparison_table.index.name = None

# Display the comparison table
print(comparison_table)