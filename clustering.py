import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/LoanData_preprocessed.csv")

# sns.scatterplot(data = df, x = 'Year', y = 'FHLBankID', hue = 'BoRace')
target=['BoRace']
X = df.loc[:, ~df.columns.isin(target)]
y = df.loc[:, df.columns.isin(target)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
from sklearn import preprocessing

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 7, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

sns.boxplot(x = kmeans.labels_, y = y_train['BoRace'])

from sklearn.metrics import silhouette_score

silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')