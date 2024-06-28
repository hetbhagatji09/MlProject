import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

from src.logger import logging
from src.utils import save_object,evaluate_model
from src.exception import customException
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('splitting training and test input data')

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]


            models={
                    'Linear Regression':LinearRegression(),
                    'Ridge':Ridge(),
                    'Lasso':Lasso(),
                    'KNeighborsRegressor':KNeighborsRegressor(),
                    'DecisionTreeRegressor':DecisionTreeRegressor(),
                    'RandomForestRegressor':RandomForestRegressor(),
                    'AdaBoostRegressor':AdaBoostRegressor(),
                    'CatboostRegressor':CatBoostRegressor(verbose=False),
                    'GradientBoosting':GradientBoostingRegressor(),
                    'XGBRegressor':XGBRegressor(),

                    }
            model_report: dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            
            ##to get a best model score from dict
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise customException('No best Model found')
            
            logging.info(f"Best model on both training and testing")
            
            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=best_model
                )
            predicted_output=best_model.predict(X_test)
            r2_score_value =r2_score(y_test,predicted_output)
            return r2_score_value
                
        except Exception as e:
            raise customException(e,sys)
