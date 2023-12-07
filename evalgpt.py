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

class GPT2Dataset(Dataset):

    def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for i in range(0, len(txt_list)):
            encodings_dict = tokenizer('<|startoftext|><query_begin>' + txt_list[i]['target'] + '<query_end><meaning_begin>' + txt_list[i]['meaning_representation'] + '<meaning_end><|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


# Get the test dataset
data = datasets.load_dataset('GEM/viggo')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
test_dataset = GPT2Dataset(data['test'], tokenizer, max_length=768)

batch_size = 2

test_dataloader = DataLoader(
            test_dataset,  # The training samples.
            sampler = RandomSampler(test_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )


# Load the pretrained model.

output_dir = '/home/rajpalleti/motivatingexp'

model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
device = torch.device("cuda")

model.to(device)
model.cuda()
# Set model to evaluation mode.
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
for i in range(len(data["test"])):

  test_prompt = '<|startoftext|><query_begin>' + data["test"][i]['target'] + '<query_end><meaning_begin>'
  # test_prompt_tokenized = torch.tokenizer(test_prompt, truncation=True, max_length=max_length, padding="max_length"))

  test_prompt_tokenized = torch.tensor(tokenizer.encode(test_prompt)).unsqueeze(0)
  test_prompt_tokenized = test_prompt_tokenized.to(device)

  test_label = data["test"][i]['meaning_representation'] + '<meaning_end><|endoftext|>'
  test_label_tokenized = torch.tensor(tokenizer.encode(test_label)).unsqueeze(0)

  test_prediction = model.generate(
                                test_prompt_tokenized.cuda(), 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=50, 
                                max_length = 300,
                                top_p=0.95, 
                                num_return_sequences=3
                                )

  # test_prediction = model.generate(
  #                                   bos_token_id=random.randint(1,30000),
  #                                   do_sample=True,   
  #                                   top_k=50, 
  #                                   max_length = 200,
  #                                   top_p=0.95, 
  #                                   num_return_sequences=1
  #                               )
  
  # print(test_label)
  # breakpoint()
  test_prediction_decoded = tokenizer.decode(test_prediction[0], skip_special_tokens=False)
  # test_prediction_meaning = test
  # breakpoint()

  meaning_begin_tag = "<meaning_begin>"
  start_index = test_prediction_decoded.find(meaning_begin_tag)
  if start_index != -1:
    predicted_meaning = test_prediction_decoded[start_index + len(meaning_begin_tag):]
    if predicted_meaning == test_label:
      num_correct_predictions += 1
      # print("Correct")

  # print()
  # breakpoint()

print(f"Accuracy = {num_correct_predictions / len(data['test'])}")

      
  



# generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
# generated = generated.to(device)

# sample_outputs = model.generate(
#                                 generated, 
#                                 #bos_token_id=random.randint(1,30000),
#                                 do_sample=True,   
#                                 top_k=50, 
#                                 max_length = 300,
#                                 top_p=0.95, 
#                                 num_return_sequences=3
#                                 )

# for i, sample_output in enumerate(sample_outputs):
#   print("{} : {} : {}\n\n".format(sample_output, i, tokenizer.decode(sample_output, skip_special_tokens=False)))