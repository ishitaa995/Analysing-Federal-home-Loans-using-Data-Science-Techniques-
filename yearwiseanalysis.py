import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, precision_recall_fscore_support
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns

df = pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/LoanData_preprocessed.csv")

ka= pd.DataFrame(columns= df['BoRace'].unique(), index=df['Year'].unique())
for i in (df['Year'].unique()):
    for j in df['BoRace'].unique():
        rrace_1_count_by_year = len(df[(df['BoRace'] == j) & (df['Year'] == i)])
        ka[j][i]= rrace_1_count_by_year

ax=ka.plot()
ax.legend(['Race 5-White', 'Race 3 - Black /African American', 'Race 7- Not Applicable','Race 2-Asian', 'Race 1- American Indian/ Alaska Native', 'Race 4 - Native Hawaiian/ other pacific Islanders','Race 6- Info not provided' ])
ax.set_xlabel('Year')
ax.set_ylabel('Count of Race')

# set the title of the graph
ax.set_title('Trends in the race of the borrower over the years')
# rrace_2_count_by_year = len(df[(df['BoRace'] == 2) & (df['Year'] == 2019)])
# rrace_3_count_by_year = len(df[(df['BoRace'] == 3) & (df['Year'] == 2019)])
# Print the result
print("Number of items with borrower race equal to 1:", rrace_1_count_by_year)

p_vals = np.zeros((len(df.columns), len(df.columns)))
for i in range(len(df.columns)):
    for j in range(i+1, len(df.columns)):
        p_val = ttest_ind(df.iloc[:,i], df.iloc[:,j])[1]
        p_vals[i][j] = p_val
        p_vals[j][i] = p_val

# create heatmap of p-values
fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(p_vals, cmap='coolwarm')

# set x and y axis labels
ax.set_xticks(np.arange(len(df.columns)))
ax.set_yticks(np.arange(len(df.columns)))
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)

# rotate x-axis labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# set title of the plot
ax.set_title("Student's T-test P-Value Matrix")

# add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('P-Value', rotation=-90, va="bottom")

# show the plot
plt.show()

corr_matrix = df.corr()
sns.heatmap(corr_matrix)
sns.heatmap(corr_matrix, cmap="YlGnBu", annot=False, linewidths=.5, cbar_kws={"shrink": .5})
plt.imshow(corr_matrix, cmap='hot', interpolation='nearest')
plt.xticks(range(len(df.columns)), df.columns)
plt.yticks(range(len(df.columns)), df.columns)
plt.xticks(rotation=90)
plt.colorbar()
plt.figure(figsize=(15, 10))
plt.subplots_adjust(bottom=0.15)
plt.show()



# plt.plot(df['Year'], label="Year")
plt.plot(df['Race2'], label="Race2")
plt.plot(df['LTV'], label="LTV")

# add a legend and labels to the plot
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Line Graph of Features')
plt.show()