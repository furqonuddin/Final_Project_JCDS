import pandas as pd
import joblib
df = pd.read_csv('diabetes_new.csv')
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter=500)
model.fit(x, y)
joblib.dump(model, 'modelML')