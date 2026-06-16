import os
from TextSummarizer.custom_logging  import logger
from TextSummarizer.entity import DataTransformationConfig
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import multiprocessing

class DataTransformation:
    def __init__(self,config):
        self.config = config
        # prefer fast tokenizer; allow network on first run to populate cache
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            use_fast=True,
            local_files_only=False,
        )
    
    def convert_to_features(self,example_batch):
        # example_batch contains lists when batched=True
        # tokenize inputs
        input_encodings = self.tokenizer(
            example_batch['dialogue'],
            max_length=1024,
            truncation=True,
        )

        # tokenize targets (summaries) separately — avoid using as_target_tokenizer()
        target_encodings = self.tokenizer(
            example_batch['summary'],
            max_length=128,
            truncation=True,
        )

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids'],
        }
    def convert(self):
        # Load already-transformed dataset (arrow) first
        transformed_dir = self.config.transformed_data_dir
        tokenized_save_path = os.path.join(self.config.root_dir, "samsum_dataset")

        # If tokenized artifacts already exist, skip tokenization
        if os.path.exists(tokenized_save_path):
            logger.info(f"Found tokenized dataset at {tokenized_save_path}, skipping tokenization")
            return

        data_samsum = load_from_disk(transformed_dir)

        # parallel, batched tokenization to speed up processing
        num_proc = max(1, (multiprocessing.cpu_count() or 1) - 1)
        data_samsum_pt = data_samsum.map(
            self.convert_to_features,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=data_samsum["train"].column_names if "train" in data_samsum else None,
            load_from_cache_file=True,
        )

        # save tokenized dataset for future runs
        data_samsum_pt.save_to_disk(tokenized_save_path)
        logger.info(f"Tokenized dataset saved to {tokenized_save_path}")
