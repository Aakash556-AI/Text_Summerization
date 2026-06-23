
from TextSummarizer.components import model_evoluation
from TextSummarizer.custom_logging import logger
from TextSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from TextSummarizer.pipeline.satge_02_data_validation import DataValidationTrainingPipeline
from TextSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from TextSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationtrainningpipeline
from streamlit import metric



stage_name = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> stage {stage_name} started <<<<<<")

    config_manager = ConfigurationManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()

    pipeline = DataIngestionTrainingPipeline(config=data_ingestion_config)
    pipeline.main()

    logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e



stage_name = "Data Validation Stage"
try:
    config_manager = ConfigurationManager()
    data_validation_config = config_manager.get_data_validation_config()
    logger.info(f">>>>>> stage {stage_name} started <<<<<<")
    data_validation = DataValidationTrainingPipeline(config=data_validation_config)
    data_validation.main()
    logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"Error in stage {stage_name}: {e}")
    raise e


stage_name = "Data Transformation Stage"
try:
    logger.info(f">>>>>> stage {stage_name} started <<<<<<")

    config_manager = ConfigurationManager()
    data_transformation_config = config_manager.get_data_transformation_config()
    data_transformation = DataTransformationTrainingPipeline(config=data_transformation_config)
    data_transformation.main()
    logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"Error in stage {stage_name}: {e}")
    raise e

stage_name = "Model Trainer Stage"
try:
    logger.info(f">>>>>> stage {stage_name} started <<<<<<")

    config_manager = ConfigurationManager()
    model_trainer_config = config_manager.get_model_trainer_config()
    model_trainer = ModelTrainerTrainingPipeline(config=model_trainer_config)
    model_trainer.main()
    logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"Error in stage {stage_name}: {e}")
    raise e


stage_name = "Model Evaluation stage"
try:
    logger.info(f">>>>>>stage{stage_name} started <<<<<<<")
    model_evoluation = ModelEvaluationtrainningpipeline()
    model_evoluation.main()
    logger.info(f"<<<<<<<stagge{stage_name} completed <<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e
    