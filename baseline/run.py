from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import math
import random
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import youtokentome as yttm
import sys
from tokenizer import tokenize
from classes import WordData
from classes import EncoderRNN_inside_class
from classes import AttentionDecoder_inside_class
from classes import My_seq2seq_attention
from clssses import train
from classes import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenized_source, tokenized_target = tokenize(sys.argv[1])

batch_size = 64
context_len = 40
target_len = 40
pad_index = 0
eos_index = 3

validation_start_index = int(len(tokenized_source) * 0.05)

train_dataset = WordData(context_list=tokenized_source[:-validation_start_index],
                         questions_list = tokenized_target[:-validation_start_index],
                         context_len=context_len, questions_len = target_len, pad_index=pad_index, eos_index=eos_index)

validation_dataset = WordData(context_list=tokenized_source[-validation_start_index:],
                              questions_list = tokenized_target[-validation_start_index:],
                         context_len=context_len, questions_len = target_len, pad_index=pad_index, eos_index=eos_index)


pad_idx = tokenizer.vocab().index("<PAD>")
eos_idx = tokenizer.vocab().index("<EOS>")
sos_idx = tokenizer.vocab().index("<BOS>")
# Size of embedding_dim should match the dim of pre-trained word embeddings!

embedding_dim = 300
hidden_dim = 300
vocab_size = len(tokenizer.vocab())
model = My_seq2seq_attention(embedding_dim,
                 hidden_dim, 
                 vocab_size, 
                 device, pad_idx, eos_idx, sos_idx).to(device)
optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=1.0e-3)
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

N_EPOCHS = 6
train_losses = []
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, criterion, optimizer, epoch)
#     print (train_loss)
    train_losses.append(train_loss)
    if min(train_losses) == train_loss and len(train_losses) > 1:
        torch.save(model.state_dict, "best_seq2seq_attention")
        torch.save(optimizer.state_dict, "best_Adam_state_dict_attention")
    
    torch.save(model.state_dict, "last_seq2seq_attention")
    torch.save(optimizer.state_dict, "Adam_state_dict_attention")
    
    #early stopping
    validation_losses = []
    test_loss = evaluate(model, validation_loader)
    validation_losses.append(test_loss)
    
    if len(validation_losses) > 1 and validation_losses[epoch] > validation_losses[epoch-1]:
        print("stop")
        break
#     break