import os
import sys  
from src.exception import customException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    #this function for read data from database
    def initiate_data_ingestion(self):
        logging.info("Entered the data_ingestion method or component.")

        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info('read the dataset as dataframe')
            ## os.path.dirname returns a path for the directory
            ##os.makedir is making directory for this path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            logging.info("Train Test split intiated")
            train_set,test_set=train_test_split(df,random_state=42,test_size=0.2)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)  

            logging.info('ingestion of the data is completed')
            return(
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path,
            )

        except Exception as e:
            raise customException(e,sys)
        
if __name__ == '__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr=data_transformation.initiate_data_transformation(train_data,test_data)
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
    

