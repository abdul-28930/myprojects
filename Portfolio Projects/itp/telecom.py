import pandas as pd
import numpy as np
import seaborn as sns
import pickle


df = pd.read_csv('churn.csv')

df.head()

# convert datatype for 'TotalCharges'
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


from sklearn.preprocessing import LabelEncoder
ohe = LabelEncoder()
df['gender']=ohe.fit_transform(df['gender'])
df['Partner']=ohe.fit_transform(df['Partner'])
df['Dependents']=ohe.fit_transform(df['Dependents'])
df['PhoneService']=ohe.fit_transform(df['PhoneService'])
df['MultipleLines']=ohe.fit_transform(df['MultipleLines'])
df['InternetService']=ohe.fit_transform(df['InternetService'])
df['OnlineSecurity']=ohe.fit_transform(df['OnlineSecurity'])
df['OnlineBackup']=ohe.fit_transform(df['OnlineBackup'])
df['DeviceProtection']=ohe.fit_transform(df['DeviceProtection'])
df['TechSupport']=ohe.fit_transform(df['TechSupport'])
df['StreamingTV']=ohe.fit_transform(df['StreamingTV'])
df['StreamingMovies']=ohe.fit_transform(df['StreamingMovies'])
df['Contract']=ohe.fit_transform(df['Contract'])
df['PaperlessBilling']=ohe.fit_transform(df['PaperlessBilling'])
df['PaymentMethod']=ohe.fit_transform(df['PaymentMethod'])

df["TotalCharges"].fillna(df["TotalCharges"].median())

# Creat independent variable and dependent variable
X = df.drop('Churn', axis=1)
y = df['Churn']


from imblearn.over_sampling import ADASYN
ada = ADASYN()
X_sample, y_sample = ada.fit_resample(X,y)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size= 0.30, random_state=101)



from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
model.fit(X_train,y_train)


pickle.dump(model,open('telecom.pkl','wb'))


