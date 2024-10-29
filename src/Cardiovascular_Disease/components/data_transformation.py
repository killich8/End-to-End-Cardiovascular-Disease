import os
from Cardiovascular_Disease.entity.config_entity import DataTransformationConfig
from Cardiovascular_Disease import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        

    def label_encode(self):
        le = LabelEncoder()
        data = pd.read_csv(self.config.data_path)
        for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
                data[col] = le.fit_transform(data[col])
        logger.info("Label encoding completed.")
        return data  
 
    
    def scale(self,data):
        mms = MinMaxScaler() # Normalization
        ss = StandardScaler() # Standardization
        data['Oldpeak'] = mms.fit_transform(data[['Oldpeak']])
        data['Age'] = ss.fit_transform(data[['Age']])
        data['RestingBP'] = ss.fit_transform(data[['RestingBP']])
        data['Cholesterol'] = ss.fit_transform(data[['Cholesterol']])
        data['MaxHR'] = ss.fit_transform(data[['MaxHR']])
        logger.info("Scaling completed.")
        return data 




    def train_test_spliting(self,data):
        

        
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)