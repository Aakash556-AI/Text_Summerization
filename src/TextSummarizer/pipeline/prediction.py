from TextSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evoluation_config()

    def predict(self,text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path,use_fast=False)
        # prediction

        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length":128}

        sample_text = dataset_samsum["test"][0]["dialogue"]
        reference = dataset_samsum["test"][0]["summary"]

        pipe = pipeline("summarization", model="pegasus-samsum-model",tokenizer=tokenizer)

        print("dialogue")
        print(sample_text)

        print("\nReference Summary:")
        print(reference)

        print("\nModel Summary:")

        return pipe
    