import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, \
    recall_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import io

df = pd.read_csv('CustomerChurn.csv')
df.head()
# convert datatype for 'TotalCharges'
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df_cat = df.select_dtypes(include=[object])
df_cat.head()
abc = df_cat.shape
print(abc)
bcd = df_cat.columns
print(bcd)
for i in df_cat.columns:
    print(i, df[i].unique())

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
ohetransform = ohe.fit_transform(df[df_cat.columns])
ohetransform.head()
df = pd.concat([df, ohetransform], axis=1)
for i in df_cat.columns:
    df.drop([i], axis=1, inplace=True)

df.describe()

# Define a custom color palette
custom_palette = sns.color_palette("Set2")

# Iterate over predictors and create count plots
for i, predictor in enumerate(df.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges', 'tenure'])):
    plt.figure(i, figsize=(5, 3))
    sns.countplot(data=df, x=predictor, hue='Churn', palette=custom_palette)

# Define a custom color palette
custom_palette = sns.color_palette("pastel")

# Iterate over predictors and create count plots
for i, predictor in enumerate(df.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges', 'tenure'])):
    plt.figure(i, figsize=(5, 3))
    sns.countplot(data=df, x=predictor, hue='Churn', palette=custom_palette)

df.head()

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
# Creat independent variable and dependent variable
X = df.drop('Churn', axis=1)
y = df['Churn']

ada = ADASYN()
X_sample, y_sample = ada.fit_resample(X, y)
print('Original dataset \n', y.value_counts())
print('Resample dataset \n', y_sample.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.30, random_state=0)

skfold = StratifiedKFold(n_splits=10)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=skfold)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(cm)
print('Accuracy Score: ', accuracy_score(y_test, y_pred))
print('')
print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print('')
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
print('')
print('Precision: {:.2f}'.format(precision))
print('')
print('Recall: {:.2f}'.format(recall))
print('')
print('F1: {:.2f}'.format(f1))
print('-----------------------------------')
print('')
