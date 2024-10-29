import os
import pandas as pd
from Cardiovascular_Disease.entity.config_entity import ModelTrainerConfig
from Cardiovascular_Disease import logger
from sklearn.linear_model import LogisticRegression
import joblib



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        lr = LogisticRegression(random_state= self.config.random_state, C=self.config.C, penalty=self.config.penalty)
        lr.fit(train_x, train_y)


        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))