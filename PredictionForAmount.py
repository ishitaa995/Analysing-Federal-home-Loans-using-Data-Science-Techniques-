import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, precision_recall_fscore_support, confusion_matrix, \
    mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


df = pd. read_csv("C:/Users/isayal/Desktop/LoanData.csv")

# Code to drop unwanted columns
df.drop(['LoanNumber', 'AssignedID','Coop','Program','AcquDate','MortDate','Bed1','Bed2','Bed3','Bed4','Aff1','Aff2','Aff3','Aff4','Occup','NumUnits','RentUt1','RentUt2','RentUt3','Product','RentUt4','Rent1','Rent2','Rent3','Rent4','FeatureID','SellType','Seller','SellCity','SellSt','CICA','LienStatus','FedFinStbltyPlan','GSEREO','FHFBID','Geog','SpcHsgGoals','AcqTyp','PrepayP'], axis=1, inplace=True)


# code to fill missing values for FHLBankID Column based on FIPSStateCode
dk = pd.DataFrame()
dk = df[['FHLBankID','FIPSStateCode']]
fill_values = dict(dk.groupby('FIPSStateCode')['FHLBankID'].apply(lambda x: x.dropna().mode()[0]))
dk['FHLBankID'] = dk.apply(lambda x: fill_values[x['FIPSStateCode']] if pd.isnull(x['FHLBankID']) else x['FHLBankID'], axis=1)
df['FHLBankID'] = dk['FHLBankID']

label_encoder = LabelEncoder()
fhlbankid_encoder = LabelEncoder()
df['FHLBankID'] = fhlbankid_encoder.fit_transform(df['FHLBankID'])

# create label encoder for PropType column
proptype_encoder = LabelEncoder()
df['PropType'] = proptype_encoder.fit_transform(df['PropType'])

cols_with_missing = df.columns[df.isnull().any()].tolist()
testdata = df[df.isnull().any(axis=1)]
df.dropna(inplace=True)
target=["Tractrat", "IncRat"]
X_train = df.loc[:, ~df.columns.isin(target)]
y_train = df.loc[:, df.columns.isin(target)]
X_test = testdata.loc[:, ~df.columns.isin(target)]
y_test = testdata.loc[:, df.columns.isin(target)]
clf = MultiOutputRegressor(RandomForestRegressor(max_depth=2, random_state=0))
clf.fit(X_train, y_train)
y_train.isnull().sum()
y_pred = clf.predict(X_test)
y_test = y_pred
X_test[target] = y_pred
X_train[target] = y_train
df = X_train.append(X_test)

# code for predictions
df.isnull().any()
target = ["Amount"]
X = df.loc[:, ~df.columns.isin(target)]
y = df.loc[:, df.columns.isin(target)]

# Define the class labels
# class_labels = {3: 0, 6: 1, 9: 2, 8: 3, 2: 4, 10: 5, 5: 6, 1: 7, 7: 8, 4: 9, 0: 10}
# class_labels = {3, 6, 9, 8, 2, 10, 5, 1, 7, 4, 0}

# Map the year values to class labels
# y['FHLBankID'] = y['FHLBankID'].map(class_labels)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=0)

# train= xgb.DMatrix(X_train,label=y_train)
# test= xgb.DMatrix(X_test,label=y_test)
#
#param={
#     'max_depth' : 4,
#     'eta': 0.3,
#     'objective':'multi:softmax',
#     'num_class':13# }
# epochs=10# model=xgb.train(param,train,epochs)
# df.info()
# label_values = (df['Year'].unique())
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=70, max_depth=3, learning_rate=0.2)

# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', =13, learning_rate=0.1, max_depth=6, n_estimators=100, colsample_bytree=0.8)

# Train the XGBoost model on the training data
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('MSE:', mse)
print('RMSE:', rmse)
print('MAE:', mae)
print('R2:', r2)


importances = xgb_model.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
# sort features by importance in descending order
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# plot feature importances
plt.barh(feature_importances['feature'], feature_importances['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()


# Make predictions on the test data
y_pred = xgb_model.predict(X_test)

# Calculate correlation between predicted and actual target values
correlation = np.corrcoef(y_pred, y_test.values.ravel())[0, 1]

print('Correlation:', correlation)


