from Cardiovascular_Disease.config.configuration import ConfigurationManager
from Cardiovascular_Disease.components.model_evaluation import ModelEvaluation
from Cardiovascular_Disease import logger


STAGE_NAME = "Model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.save_results()