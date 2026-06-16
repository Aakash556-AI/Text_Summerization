from TextSummarizer.constants import *
from TextSummarizer.utils.common import read_yaml, create_directories

from TextSummarizer.entity import (DataIngestionConfig, DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvoluationConfig)
from dataclasses_json import config


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([Path(config.root_dir)])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            dataset_download_url=config.dataset_download_url,
            raw_data_dir=Path(config.raw_data_dir),
            local_data_file=Path(config.local_data_file),
            unzipped_data_dir=Path(config.unzipped_data_dir),
        )

        return data_ingestion_config
    


    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([Path(config.root_dir)])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=Path(config.STATUS_FILE),
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([Path(config.root_dir)])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            transformed_data_dir=Path(config.transformed_data_dir),
            tokenizer_name=config.tokenizer_name,
        )
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        model_trainer_config = self.config.model_trainer
        training_params = self.params.TrainingArguments
        create_directories([Path(model_trainer_config.root_dir)])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(model_trainer_config.root_dir),
            data_path=Path(model_trainer_config.data_path),
            model_ckpt_dir=Path(model_trainer_config.model_ckpt_dir),
            num_train_epochs=training_params.num_train_epochs,
            warmup_steps=training_params.warmup_steps,
            per_device_train_batch_size=training_params.per_device_train_batch_size,
            per_device_eval_batch_size=training_params.per_device_eval_batch_size,
            weight_decay=training_params.weight_decay,
            logging_steps=training_params.logging_steps,
            save_steps=training_params.save_steps,
            gradient_accumulation_steps=training_params.gradient_accumulation_steps,
            report_to=training_params.report_to,
        )

        return model_trainer_config

    def get_model_evoluation_config(self) -> ModelEvoluationConfig:
    
        config = self.config.model_evoluation
        create_directories([config.root_dir])

        model_evoluation_config = ModelEvoluationConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            model_path = config.model_path,
            tokenizer_path = config.tokenizer_path,
            matric_file_name = config.matric_file_name
    )

        return model_evoluation_config