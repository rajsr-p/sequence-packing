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
# output_dir = '/home/rajpalleti/simplesequencepacking'
# output_dir = '/home/DanielKim/simplesequencepacking_ablation1'
output_dir = '/home/rehaan/sequence-packing/model_weights/uhohtest1'

model = GPT2LMHeadModel.from_pretrained(output_dir)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
# tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<query_begin>", "<query_end>", "<meaning_begin>", "<meaning_end>"]
})
# model.resize_token_embeddings(len(tokenizer))

# breakpoint()

device = torch.device("cuda")

model.to(device)
model.cuda()
model.eval()


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
path = "simple_sequence_packing_ablation2_output.txt"
 
for i in range(len(data["test"])):

  # test_prompt = '<query_begin>' + data["test"][i]['target'] + '<query_end><meaning_begin>'
  test_prompt = '<|startoftext|><query_begin>' + data["test"][i]['target'] + '<query_end><meaning_begin>'
  # test_prompt_tokenized = torch.tokenizer(test_prompt, truncation=True, max_length=max_length, padding="max_length"))

  test_prompt_tokenized = torch.tensor(tokenizer.encode(test_prompt)).unsqueeze(0)
  test_prompt_tokenized = test_prompt_tokenized.to(device)

  test_label = data["test"][i]['meaning_representation'] + '<meaning_end><|endoftext|>'
  # test_label = data["test"][i]['meaning_representation'] + '<meaning_end>'
  # test_label_tokenized = torch.tensor(tokenizer.encode(test_label)).unsqueeze(0)

  test_prediction = model.generate(
                                test_prompt_tokenized.cuda(), 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=False,   
                                top_k=50, 
                                max_length = 300,
                                top_p=0.95, 
                                num_return_sequences=1
                                )
  
  
  # print(test_label)
  # breakpoint()
  test_prediction_decoded = tokenizer.decode(test_prediction[0], skip_special_tokens=False)
  # test_prediction_meaning = test
  # breakpoint()

  meaning_begin_tag = "<meaning_begin>"
  start_index = test_prediction_decoded.find(meaning_begin_tag)

  if start_index != -1:
    predicted_meaning = test_prediction_decoded[start_index + len(meaning_begin_tag):]

    with open(path, 'a') as file: 
      file.write(predicted_meaning + '\n')
      file.write(test_label + '\n')

      if predicted_meaning == test_label:
        num_correct_predictions += 1
        file.write('\n')
        # print("Correct")
      else: 
        file.write("ABOVE MARKED INCORRECT\n\n")
  else: 
    with open(path, 'a') as file: 
      file.write("<meaning_begin> was not found in the decoded test prediction.\n")
      file.write("ABOVE MARKED INCORRECT\n\n")
  
  # breakpoint()

  # print()
  # breakpoint()

print(f"Accuracy = {num_correct_predictions / len(data['test'])}")
