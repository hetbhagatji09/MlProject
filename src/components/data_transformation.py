import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import customException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.Data_Transformation_Config=DataTransformationConfig()

    def Get_Data_Transformer_Object(self):
        """This function is responsable for data transformation"""
        try:
            numerical_features=['reading_score',
                                'writing_score']
            categorical_features=['gender',
                                  'race_ethnicity',
                                  'parental_level_of_education',
                                  'lunch',
                                  'test_preparation_course'
                                ]
            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())

                ]
            )
            cat_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical Columns{categorical_features}")
            logging.info(f"Numerical Columns{numerical_features}")

            preproessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_features),
                    ("cat_pipeline",cat_pipeline,categorical_features)
                ]
            )
            return preproessor
            
        except Exception as e:
            raise customException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data is completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.Get_Data_Transformer_Object()
            target_column_name='math_score'
            numeric_columns=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe .")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
        
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info(f"Saved preprocessing object.") 

            save_object(
                file_path=self.Data_Transformation_Config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )
            return(
                train_arr,
                test_arr,
                # self.Data_Transformation_Config.preprocessor_obj_file_path
            
            )               
    
        except Exception as e:
            raise customException(e,sys)