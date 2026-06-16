
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_from_disk
from zmq import device
import evaluate
import torch
import pandas as pd

from TextSummarizer.entity import ModelEvoluationConfig



class ModelEvoluation:
    def __init__(self, config: ModelEvoluationConfig):
        self.config = config
    # Evaluation

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(
    self,
    dataset,
    metric,
    model,
    tokenizer,
    column_text="dialogue",
    column_summary="summary",
    batch_size=16,
    device=None
):
        # Determine device if not provided
        if device is None:
            device = "mps" if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")

        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size
                                                                ))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        from tqdm import tqdm

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)
        ):

            inputs = tokenizer(article_batch, max_length=256,truncation=True,
                        padding="max_length", return_tensors="pt")

            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                          attention_mask= inputs["attention_mask"].to(device),
                          length_penalty=0.8, num_beams=8, max_length=128)
            ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''
            # finally ,we decode the  generated texts,
            # replace the token and add the decoded texts with the referance to the matric.
            decoded_summaries= [tokenizer.decode(s,skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
                                for s in summaries]

            metric.add_batch(predictions=decoded_summaries, references=target_batch)
# finally compute and return the rouge scores


        score = metric.compute()
        return score
    
    def evaluate(self):
       device = "mps" if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
       tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path,use_fast=False)
       model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
       dataset = load_from_disk(self.config.data_path)["test"]
       print(self.config.tokenizer_path)


       rouge_name = "rouge1", "rouge2", "rougeL", "rougeLsum"
       metric = evaluate.load("rouge", rouge_types=rouge_name)
       score = self.calculate_metric_on_test_ds(dataset, metric, model, tokenizer, device=device)
       rouge_dict = {k: round(v*100, 4) for k,v in score.items()}

       df = pd.DataFrame(rouge_dict, index=[0])
       df.to_csv(self.config.matric_file_name, index=False)
    