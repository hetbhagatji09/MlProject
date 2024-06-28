import os
import sys
import numpy as np
import pandas as pd
from src.exception import customException
from src.logger import logging
import dill

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            #here our obj is saved in this file path
    except Exception as e:
        raise customException(e,sys)
