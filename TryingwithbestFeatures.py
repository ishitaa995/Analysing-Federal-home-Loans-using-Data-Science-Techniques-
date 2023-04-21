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
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
df = pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/LoanData_preprocessed.csv")
dk= df[['MinPer', 'LTV', 'First', 'BoGender', 'CoAge', 'Rate', 'Self', 'PropType', 'ArmMarg', 'BoEth','BoRace']]

target=["BoRace"]
X=df.loc[:,~df.columns.isin(target)]
y=df.loc[:,df.columns.isin(target)]
feature_names=X.columns
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
scaler = MinMaxScaler(feature_range=(0, 1))

selector = SelectKBest(chi2, k=10)
X_new = selector.fit_transform(X_scaled_df, y)
# print selected features
selected_features=selector.get_support(indices=True)
print([feature_names[i] for i in selected_features])
# """"""""uncomment for year classification""""
# Define the class labels
# class_labels = {2009: 0, 2010: 1, 2011: 2, 2012: 3, 2013: 4, 2014: 5, 2015: 6, 2016: 7, 2017: 8, 2018: 9, 2019: 10, 2020: 11, 2021: 12}

class_labels = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
# y[target] = y[target].map(class_labels)
# Map the year values to class labels
y[target] = y[target].applymap(class_labels.get)

X_train, X_test, y_train, y_test = train_test_split( X_scaled_df, y, test_size=0.30, random_state=0)

# train= xgb.DMatrix(X_train,label=y_train)
# test= xgb.DMatrix(X_test,label=y_test)
#
# param={
#     'max_depth' : 4,
#     'eta': 0.3,
#     'objective':'multi:softmax',
#     'num_class':13
# }
# epochs=10
# model=xgb.train(param,train,epochs)
# df.info()
# label_values = (df['Year'].unique())

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=13, learning_rate=0.1, max_depth=6, n_estimators=100, colsample_bytree=0.8)
# Train the XGBoost model on the training data
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
dm=pd.DataFrame()
dm['Actual']=y_test
dm['Predicted']=y_pred

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

# Assuming y_pred and y_true are the predicted and true labels respectively
cm = confusion_matrix(y_test, y_pred)
# classes=(df['Year'].unique())
classes=(df['BoRace'].unique())
    # .sort()

type(cm)
# Print the confusion matrix
print(cm)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.annotate(str(cm[i][j]), xy=(j, i), ha='center', va='center')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label of Borrower race')
plt.ylabel('True Label of Borrower race')
plt.tight_layout()
plt.show()
# from sklearn.metrics import plot_confusion_matrix
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(figsize=(8, 8))
# plot_confusion_matrix(xgb_model, X_test, y_test,
#                       cmap=plt.cm.Blues,
#                       display_labels=['1', '2', '3', '4', '5', '6', '7'],
#                       ax=ax)
# plt.title('Confusion Matrix')
# plt.show()

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

similarity_matrix = cm / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, annot=True, cmap='Blues')
plt.title('Similarity Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

dx = pd.DataFrame(X_test, columns=X_test.columns)
dx["predicted_value"] = y_pred

corr_matrix = dx.corr()
plt.imshow(corr_matrix, cmap='hot', interpolation='nearest')
plt.xticks(range(len(dx.columns)), dx.columns)
plt.yticks(range(len(dx.columns)), dx.columns)
plt.xticks(rotation=90)
plt.colorbar()
plt.figure(figsize=(15, 10))
plt.subplots_adjust(bottom=0.15)
plt.show()




dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
y_pred=dummy_clf.predict(X_test)
dummy_clf.score(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print("Accuracy:", accuracy)




