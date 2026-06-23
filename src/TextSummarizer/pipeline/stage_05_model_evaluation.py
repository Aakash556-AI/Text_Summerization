from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.model_evoluation import ModelEvoluation
from TextSummarizer.custom_logging import logger


class ModelEvaluationtrainningpipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        model_eval_config = config.get_model_evoluation_config()

        model_eval = ModelEvoluation(
            config=model_eval_config
        )

        model_eval.evaluate()

        logger.info("Model Evaluation Completed")