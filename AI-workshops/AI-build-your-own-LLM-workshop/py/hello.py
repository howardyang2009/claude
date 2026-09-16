# print("Hello, world!")

#@title 0) setup
import logging
import warnings
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./gpt2")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

print(model)

from torchinfo import summary

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2")
input_ids = tokenizer.encode("Hello World!", return_tensors='pt')

# Visualize the model structure with input data to see output shapes
summary(model, input_data=input_ids)