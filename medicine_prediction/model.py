import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import numpy as np
df = pd.read_csv("hello.csv")
df.head()
X = df.drop(['output'],axis='columns')
X.head()
y = df['output']
y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=50)
model = RandomForestClassifier()
model.fit(X_train,y_train)
model.score(X_train,y_train)
model.score(X_test,y_test)
model.predict([[0,0,1,1,147,238]])
predictions = model.predict(X_test)
accuracy = model.score(X_test,y_test)
new_features = [[0,0,1,1,147,238]]
predicted_result = model.predict(new_features)
print("Predicted Result:",predicted_result)
import pickle
pickle.dump(model, open("model.pkl", "wb"))