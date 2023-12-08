import os
import time
import datetime

import pandas as pd
import seaborn as sns
import numpy as np
import random

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
import datasets

# Get the test dataset
data = datasets.load_dataset('GEM/viggo')

# Load the pretrained model.

# output_dir = '/home/rajpalleti/motivatingexp'
output_dir = '/home/rajpalleti/simplesequencepacking'

model = GPT2LMHeadModel.from_pretrained(output_dir)

model_original = GPT2LMHeadModel.from_pretrained('/home/rajpalleti/motivatingexp')

tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
device = torch.device("cuda")

model.to(device)
model.cuda()
# Set model to evaluation mode.
model.eval()

model_original.to(device)
model_original.cuda()
model_original.eval()



seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# breakpoint()
# prompt = data[test]

# breakpoint()
# for example in data[test]

num_correct_predictions = 0
for i in range(len(data["test"])):

  test_prompt = '<|startoftext|><query_begin>' + data["test"][i]['target'] + '<query_end><query_meaning_separator><meaning_begin>'
  # test_prompt_tokenized = torch.tokenizer(test_prompt, truncation=True, max_length=max_length, padding="max_length"))

  test_prompt_tokenized = torch.tensor(tokenizer.encode(test_prompt)).unsqueeze(0)
  test_prompt_tokenized = test_prompt_tokenized.to(device)

  test_label = data["test"][i]['meaning_representation'] + '<meaning_end><|endoftext|>'
  # test_label_tokenized = torch.tensor(tokenizer.encode(test_label)).unsqueeze(0)

  test_prediction = model.generate(
                                test_prompt_tokenized.cuda(), 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=50, 
                                max_length = 300,
                                top_p=0.95, 
                                num_return_sequences=3
                                )
  
  test_prediction_original = model_original.generate(
                                test_prompt_tokenized.cuda(), 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=50, 
                                max_length = 300,
                                top_p=0.95, 
                                num_return_sequences=3
                                )
  
  # print(test_label)
  # breakpoint()
  test_prediction_decoded = tokenizer.decode(test_prediction[0], skip_special_tokens=False)
  test_prediction_decoded_original = tokenizer.decode(test_prediction_original[0], skip_special_tokens=False)
  # test_prediction_meaning = test
  # breakpoint()

  meaning_begin_tag = "<meaning_begin>"
  start_index = test_prediction_decoded.find(meaning_begin_tag)

  start_index_original = test_prediction_decoded_original.find(meaning_begin_tag)
  if start_index != -1:
    predicted_meaning = test_prediction_decoded[start_index + len(meaning_begin_tag):]
    predicted_meaning_original = test_prediction_decoded_original[start_index_original + len(meaning_begin_tag):]
    if predicted_meaning == test_label:
      num_correct_predictions += 1
      # print("Correct")
  
  breakpoint()

  # print()
  # breakpoint()

print(f"Accuracy = {num_correct_predictions / len(data['test'])}")
