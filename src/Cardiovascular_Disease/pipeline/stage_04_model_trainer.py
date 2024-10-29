from Cardiovascular_Disease.config.configuration import ConfigurationManager
from Cardiovascular_Disease.components.model_trainer import ModelTrainer
from Cardiovascular_Disease import logger
from pathlib import Path


STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()