import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json 
#load the iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')  
#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)      
#save the trained model to a file
joblib.dump(model, 'model.joblib')    
#Calculate and save metrix  
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
metrics = {
    'train_Accuracy': float(train_score),
    'test_Accuracy': float(test_score)
}   
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)   
    print(f"training_Accuracy: {train_score:.4f}")
    print(f"testing_Accuracy: {test_score:.4f}")    