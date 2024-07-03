import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import customException
from src.logger import logging
import dill
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            #here our obj is saved in this file path
    except Exception as e:
        raise customException(e,sys)
def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        
        
        for i in range(len(list(models))):
            with mlflow.start_run():
            
                model=list(models.values())[i]
                para=param[list(models.keys())[i]]
                gs=GridSearchCV(model,para,cv=3)
            
                gs.fit(X_train,y_train)
            
                best_params = gs.best_params_
                print("Best parameters for model {}: {}".format(list(models.keys())[i], best_params))
                model.set_params(best_params)
      
                model.fit(X_train,y_train)
                y_train_pred=model.predict(X_train)
                y_test_pred=model.predict(X_test)
                train_model_score=r2_score(y_train,y_train_pred)
                test_model_score=r2_score(y_test,y_test_pred)
                mlflow.log_param("best param",best_params[0])
                mlflow.log_metric("r2_score",test_model_score)
                
                remote_server_uri = "C:\\Users\\hetbh\\OneDrive\\Desktop\\MlProject\\mlruns"
                mlflow.set_tracking_uri(remote_server_uri)

                
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="Best_Model"
                )
                
            
                report[list(models.keys())[i]]=test_model_score
        return report
    
    except Exception as e:
        raise customException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise customException(e,sys)
    
    
    
    
        