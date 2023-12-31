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

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

# import nltk
# nltk.download('punkt')

import datasets
data = datasets.load_dataset('GEM/viggo')
output_dir = '/home/rehaan/sequence-packing/model_weights/uhohtest9'

os.mkdir(output_dir)
file1 = open(output_dir + "/training.txt", "a")  # append mode
file1.write(str("begin test") +  "\n")
file1.close()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium

tokenizer.add_special_tokens({
    "additional_special_tokens": ["<query_begin>", "<query_end>", "<meaning_begin>", "<meaning_end>"]
})

# print("Meaning Representation: ", data['train'][0]['meaning_representation'])
# print("Target: ", data['train'][0]['target'])

# kept here for reference 
# for i in range(0, len(data['train'])):
#     print(data['train'][i])
#     print(data['train'][i]['meaning_representation'])
#     breakpoint()
# for elem in data['train']:
#     print(elem['meaning_representation'])
#     print(elem['target'])
#     breakpoint()

SEQUENCE_PACK = True
batch_size = 4
SEQ_LENGTH = 768 # 160 or 768

class GPT2Dataset(Dataset):

    def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        # for i in range(0, len(txt_list)):
        #     encodings_dict = tokenizer('<|startoftext|><query_begin>' + txt_list[i]['target'] + '<query_end><meaning_begin>' + txt_list[i]['meaning_representation'] + '<meaning_end><|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
        #     self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
        #     self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

        whole_list = []

        i = 0
        while i < len(txt_list):            
            encodings_dict = tokenizer('<|startoftext|><query_begin>' + txt_list[i]['target'] + '<query_end><meaning_begin>' + txt_list[i]['meaning_representation'] + '<meaning_end><|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            encodings_dict['token_len_list'] = [len(tokenizer('<|startoftext|><query_begin>' + txt_list[i]['target'] + '<query_end><meaning_begin>' + txt_list[i]['meaning_representation'] + '<meaning_end><|endoftext|>')['input_ids'])]

            whole_list.append(len(tokenizer('<query_begin>' + txt_list[i]['target'] + '<query_end><meaning_begin>' + txt_list[i]['meaning_representation'] + '<meaning_end>')['input_ids']))

            j = i + 1
            while j < len(txt_list):
                if SEQUENCE_PACK == False:
                    break
                cur_sequence = ''
                tok_len = 0
                token_lens = []
                for k in range(i, j + 1):
                    new_sequence = '<|startoftext|><query_begin>'
                    new_sequence += txt_list[k]['target']
                    new_sequence += '<query_end>'
          
                    new_sequence += '<meaning_begin>'
                    new_sequence += txt_list[k]['meaning_representation']
                    new_sequence += '<meaning_end><|endoftext|>'
                    new_tok_len = len(tokenizer(new_sequence)['input_ids'])
                    tok_len += new_tok_len
                    
                    token_lens.append(new_tok_len)

                    cur_sequence += new_sequence
                
                # cur_sequence += '<|endoftext|>'
                
                encodings_dict_2 = tokenizer(cur_sequence, truncation=True)
                if len(encodings_dict_2['input_ids']) > max_length:
                    # print("Too Long Sequence: ", cur_sequence)
                    # print("Length of Too Long Encoding: ", len(encodings_dict_2['input_ids']))
                    # print("Final Length: ", len(encodings_dict['input_ids']))
                    # print("Final Token IDs: ", encodings_dict['input_ids'])
                    # print("------------------")

                    break
                encodings_dict = tokenizer(cur_sequence, truncation=True, max_length=max_length, padding="max_length")
                encodings_dict['token_len_list'] = token_lens
                # breakpoint()
                j += 1
            
            # print(encodings_dict['token_len_list'])
            existing_mask = torch.tensor(encodings_dict['attention_mask'])
            seq_len = existing_mask.shape[0]
            new_mask = torch.zeros([seq_len, seq_len])
            
            # print(encodings_dict['token_len_list'])

            prev_mask_len = 0
            for elem in encodings_dict['token_len_list']:
                new_mask[prev_mask_len:prev_mask_len+elem, prev_mask_len:prev_mask_len+elem] = 1
                prev_mask_len += elem

            # new_mask[0:prev_mask_len+1, 0] = 1
            # new_mask[0, 0:prev_mask_len+1] = 1
            # new_mask[prev_mask_len, 0:prev_mask_len+1] = 1
            # new_mask[0:prev_mask_len+1, prev_mask_len] = 1
            sumlen = int(torch.sum(new_mask[0]))
            # new_mask[:, :sumlen] = 1
            # breakpoint()

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(new_mask)

            i = j     
        
        # breakpoint()
        
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 


dataset = GPT2Dataset(data['train'], tokenizer, max_length=SEQ_LENGTH) # 160 or 768

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# I'm not really doing anything with the config buheret
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
model.cuda()

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# some parameters I cooked up that work reasonably well
# epochs = 5
epochs = 5

learning_rate = 2e-3
# learning_rate = 2e-3
warmup_steps = 10 #if SEQUENCE_PACK else 100
epsilon = 1e-8

# this produces sample output every 100 steps
sample_every = 100

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

total_t0 = time.time()
training_stats = []
model = model.to(device)

import time

for epoch_i in range(0, epochs):
    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0
    model.train()

    padded_spots = 0
    start = time.time()

    for step, batch in enumerate(train_dataloader):
        if step % 10 == 0:
            print("step: " + str(step))

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        seq_length = b_masks.shape[-1]

        new_mask = (b_masks.sum(dim=1) > 0)
        padded_spots += sum(SEQ_LENGTH - torch.sum(new_mask, dim=1))
        # print("pre-breakpoint 1")
        # breakpoint()
        
        model.zero_grad()        
        outputs = model(  b_input_ids,
                          labels=b_labels,
                          output_attentions=True, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )

        # print("outputs done, check outputs.attentions")
        # breakpoint()
        loss = outputs[0]  

        batch_loss = loss.item()
        total_train_loss += batch_loss

        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

            # model.eval()

            # sample_outputs = model.generate(
            #                         bos_token_id=random.randint(1,30000),
            #                         do_sample=True,   
            #                         top_k=50, 
            #                         max_length = 200,
            #                         top_p=0.95, 
            #                         num_return_sequences=1
            #                     )
            # for i, sample_output in enumerate(sample_outputs):
            #       print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            
            # model.train()

        loss.backward()
        optimizer.step()
        scheduler.step()

    end = time.time()
    print("end - start: " + str(end-start))
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
    print(" padded spots: " + str(padded_spots))   
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        with torch.no_grad():        

            outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                            attention_mask = b_masks,
                            labels=b_labels)
          
            loss = outputs[0]  
            
        batch_loss = loss.item()
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)    

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    file1 = open(output_dir + "/training.txt", "a")  # append mode
    file1.write(str(avg_train_loss) +  "\n")
    file1.close()

    file1 = open(output_dir + "/validation.txt", "a")  # append mode
    file1.write(str(avg_val_loss) + "\n")
    file1.close()

    file1 = open(output_dir + "/padded_spots.txt", "a")  # append mode
    file1.write(str(padded_spots) + "\n")
    file1.close()

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )


print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# output_dir = '/home/DanielKim/simplesequencepacking_ablation2'
# output_dir = '/home/rajpalleti/simplesequencepacking_epochs=10_lr=2e-3'
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)