import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, RocCurveDisplay, classification_report, precision_recall_curve
from urllib.parse import urlparse
import numpy as np
import joblib
from Cardiovascular_Disease.entity.config_entity import ModelEvaluationConfig
from Cardiovascular_Disease.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self, actual, pred):
        accuracy=accuracy_score(actual,pred)
        roc=roc_auc_score(actual,pred)
        print("Accuracy : ",'{0:.2%}'.format(accuracy))
        print("ROC_AUC Score : ",'{0:.2%}'.format(roc))
     

        return accuracy, roc
    


    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_qualities = model.predict(test_x)
        RocCurveDisplay.from_estimator(model, test_x, test_y)
        plt.title('ROC_AUC_Plot')
        plt.show()
        (accuracy, roc) = self.eval_metrics(test_y, predicted_qualities)
        
        # Saving metrics as local
        scores = {"accuracy": accuracy, "roc": roc}
        save_json(path=Path(self.config.metric_file_name), data=scores)