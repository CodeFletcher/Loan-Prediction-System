import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle
import json

df = pd.read_csv('Datasets/processed_data.csv')
df = df.drop(columns=['Unnamed: 0', 'PROSPECTID'])
y = df['Approved_Flag']
X = df.drop(columns=['Approved_Flag'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

trf1 = ColumnTransformer([
    ('or_education', OrdinalEncoder(categories=[['OTHERS', 'SSC', '12TH', 'UNDER GRADUATE', 'GRADUATE', 'POST-GRADUATE', 'PROFESSIONAL']]), [47]),
    ('ohe', OneHotEncoder(drop='first',sparse_output=False, handle_unknown='ignore'), [46, 49, 63, 64])
], remainder='passthrough')

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

trf2 = XGBClassifier(
                        objective='multi:softmax', 
                        num_class=4,
                        colsample_bytree=0.3,
                        learning_rate=0.1,
                        max_depth=5,
                        alpha=1,
                        n_estimators=100
                    )

pipe = Pipeline([
    ('trf1', trf1),
    ('trf2', trf2)
])

pipe.fit(X_train, y_train)
pickle.dump(pipe, open('model.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))
with open('feature_columns.json', 'w') as f:
    json.dump(list(X.columns), f)