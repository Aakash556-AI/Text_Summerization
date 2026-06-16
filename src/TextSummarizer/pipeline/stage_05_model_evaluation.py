from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.model_evoluation import ModelEvoluation
from TextSummarizer.custom_logging import logger

class ModelEvaluationtrainningpipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evoluation_config = config.get_model_evoluation_config()
        model_evoluation = ModelEvoluation(config=model_evoluation_config)
        model_evoluation.evaluate()