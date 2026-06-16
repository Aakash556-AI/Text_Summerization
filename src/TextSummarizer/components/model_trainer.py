from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from datasets import load_from_disk
from TextSummarizer.custom_logging import logger
from TextSummarizer.entity import ModelTrainerConfig
import torch
import os
from pathlib import Path
import types


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # ---------------- Device ----------------
        # Disable MPS at runtime to avoid unstable MPS allocations during training
        # (some PyTorch builds/macOS combos show low-watermark or OOM failures).
        try:
            if getattr(torch.backends, "mps", None):
                torch.backends.mps.is_available = lambda: False
        except Exception:
            # if we cannot override, continue and let normal detection run
            logger.debug("Could not override torch.backends.mps.is_available")

        # Prefer MPS on Apple Silicon if available, otherwise CUDA, otherwise CPU.
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # For reliability on developer machines, prefer CPU if MPS is present but
        # we disabled it above; log the final selection.
        logger.info(f"Using device: {self.device}")

        # ---------------- Model & Tokenizer ----------------
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_ckpt_dir,
            use_fast=True,
            local_files_only=False,  # allow initial download if needed
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt_dir,
        
            local_files_only=False,
        )

        # enable gradient checkpointing to reduce memory on MPS
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            logger.info("gradient checkpointing not available for this model")

        # Move model to device but catch known MPS errors (OOM / low-watermark)
        try:
            model = model.to(self.device)
        except RuntimeError as exc:
            msg = str(exc)
            if "MPS backend out of memory" in msg or "low watermark" in msg or "watermark" in msg:
                logger.warning(f"Failed to move model to {self.device}: {exc}. Falling back to CPU.")
                self.device = "cpu"
                model = model.to(self.device)
            else:
                raise

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model
        )

        # ---------------- Dataset ----------------
        tokenized_dir = os.path.join(self.config.root_dir, "samsum_dataset")

        def _tokenize_and_save(raw_dataset, target_dir):
            def preprocess_function(batch):
                # shorter max_length to reduce memory footprint on CPU
                inputs = tokenizer(
                    batch["dialogue"],
                    truncation=True,
                    padding="longest",
                    max_length=384,
                )
                # Tokenize targets separately (avoid as_target_tokenizer)
                # shorter target length to reduce memory and speed up training
                labels = tokenizer(
                    batch["summary"],
                    truncation=True,
                    padding="longest",
                    max_length=96,
                )

                inputs["labels"] = labels["input_ids"]
                return inputs

            num_proc = max(1, (os.cpu_count() or 1) - 1)
            # smaller batch size for tokenization to limit memory spikes on CPU
            tokenized_dataset = raw_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=256,
                num_proc=num_proc,
                remove_columns=raw_dataset["train"].column_names,
                load_from_cache_file=True,
            )
            tokenized_dataset.save_to_disk(target_dir)
            logger.info(f"Saved tokenized dataset to {target_dir}")
            return tokenized_dataset

        tokenized_dataset = None
        # Try loading tokenized dataset from model_trainer path
        if os.path.exists(tokenized_dir):
            try:
                logger.info(f"Loading tokenized dataset from {tokenized_dir}")
                tokenized_dataset = load_from_disk(tokenized_dir)
            except Exception as e:
                logger.warning(f"Existing path {tokenized_dir} not a HF dataset: {e}")

        # If not found, try data_transformation artifacts location under artifacts_root
        if tokenized_dataset is None:
            try:
                artifacts_root = Path(self.config.root_dir).parent
                candidate = artifacts_root / "data_transformation" / "samsum_dataset"
                if candidate.exists():
                    logger.info(f"Trying tokenized dataset at {candidate}")
                    tokenized_dataset = load_from_disk(str(candidate))
            except Exception as e:
                logger.warning(f"Could not load tokenized dataset from data_transformation: {e}")

        # If still None, try loading raw dataset from configured data_path; if that is a Dataset, tokenize and save
        if tokenized_dataset is None:
            try:
                logger.info(f"Loading raw dataset from {self.config.data_path}")
                raw_dataset = load_from_disk(str(self.config.data_path))
                tokenized_dataset = _tokenize_and_save(raw_dataset, tokenized_dir)
            except Exception as e:
                logger.exception(f"Failed to load and tokenize dataset from {self.config.data_path}: {e}")
                raise

        logger.info(
            f"Dataset columns after tokenization: {tokenized_dataset['train'].column_names}"
        )

        # ---------------- Training Arguments ----------------
        def build_training_args(train_batch=None, eval_batch=None):
            # choose dataloader workers: 0 on CPU to avoid multiprocessing pickling issues
            if getattr(self, "device", None) == "cpu":
                dataloader_workers = 0
            else:
                dataloader_workers = max(1, (os.cpu_count() or 1) - 1)

            return TrainingArguments(
                output_dir=str(self.config.root_dir),
                num_train_epochs=self.config.num_train_epochs,
                # reduce default batch sizes on CPU to avoid memory pressure
                per_device_train_batch_size=(train_batch if train_batch is not None else (max(1, int(self.config.per_device_train_batch_size // 2)) if getattr(self, "device", None) == "cpu" else self.config.per_device_train_batch_size)),
                per_device_eval_batch_size=(eval_batch if eval_batch is not None else (max(1, int(self.config.per_device_eval_batch_size // 2)) if getattr(self, "device", None) == "cpu" else self.config.per_device_eval_batch_size)),
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                logging_steps=self.config.logging_steps,
                save_steps=int(self.config.save_steps),
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                report_to=self.config.report_to,
                eval_strategy="steps",
                # Remove columns that are not model inputs (e.g., string 'id') so the
                # data collator doesn't attempt to convert them to tensors.
                remove_unused_columns=True,
                dataloader_num_workers=dataloader_workers,
                # keep only a small number of checkpoints on disk
                save_total_limit=2,
                gradient_checkpointing=True,
            )

        training_args = build_training_args()

        # ---------------- Trainer ----------------
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
        )

        # ---------------- Train ----------------
        try:
            trainer.train()
        except RuntimeError as exc:
            # Handle MPS OOM or low-watermark failures during training by
            # retrying on CPU with a reduced batch size once.
            msg = str(exc)
            if "MPS backend out of memory" in msg or "low watermark" in msg or "watermark" in msg:
                logger.warning(f"Trainer.train() failed on {self.device}: {exc}. Retrying on CPU with smaller batch sizes.")

                # reduce batch sizes (at least 1)
                new_train_bs = max(1, int(self.config.per_device_train_batch_size // 2))
                new_eval_bs = max(1, int(self.config.per_device_eval_batch_size // 2))

                # move model to CPU
                try:
                    model = model.to("cpu")
                except Exception:
                    pass

                # rebuild training args with smaller batch sizes
                training_args = build_training_args(train_batch=new_train_bs, eval_batch=new_eval_bs)

                # recreate trainer on CPU
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=tokenized_dataset["train"],
                    eval_dataset=tokenized_dataset["validation"],
                )

                # Prevent Trainer/accelerate from attempting to move the model back to MPS
                # by monkeypatching the instance method that moves the model to device.
                def _move_model_to_device(self, model_, device_, args=None):
                    try:
                        return model_.to("cpu")
                    except Exception:
                        return model_

                trainer._move_model_to_device = types.MethodType(_move_model_to_device, trainer)

                trainer.train()
            else:
                raise

        # ---------------- Save ----------------
        model.save_pretrained(
            os.path.join(self.config.root_dir, "model_pegasus")
        )
        tokenizer.save_pretrained(
            os.path.join(self.config.root_dir, "tokenizer")
        )

        logger.info("Model and tokenizer saved successfully.")
