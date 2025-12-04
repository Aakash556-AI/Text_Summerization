
from TextSummarizer.custom_logging import logger
from TextSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from TextSummarizer.pipeline.satge_02_data_validation import DataValidationTrainingPipeline
from TextSummarizer.config.configuration import ConfigurationManager

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
    data_validation = DataValidationTrainingPipeline(config=None)
    data_validation.main()
    logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"Error in stage {stage_name}: {e}")
    raise e
