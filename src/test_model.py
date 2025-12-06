import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import joblib
import json 
import pytest   

def test_model_prediction():
    # Load the iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')  
    
    # Load the trained model
    model = joblib.load('model.joblib')

    #make predictions   
    prediction = model.predict(X)

    #Basic validation of predictions
    assert len(prediction) == len(y), "predictions length does not match traget length!"

def test_model_accuracy():
    # Load the iris dataset   
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)  
    y = pd.Series(iris.target, name='target')
    # Load the trained model
    model = joblib.load('model.joblib') 

    #Check accuracy
    accuracy = model.score(X, y)
    assert accuracy >= 0.8, f"Model accuracy {accuracy} is below 0.8"    