import os
os.system('pip install tqdm')
os.system('pip install pymorphy2')
os.system('pip install torchtext')
os.system('pip install git+https://github.com/aatimofeev/spacy_russian_tokenizer.git')

import torch
import torch.nn as nn
import torch.optim as optim
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS
from spacy.lang.ru import Russian
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import torchtext
import net
import spacy
import random
import math
import time
from tqdm import tqdm

nlp = Russian()
russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)

TEXT = torchtext.data.Field(tokenize = net.tokenize_ru, lower=True, init_token = '<sos>', 
            eos_token = '<eos>', fix_length = 20)
fields = [('src', TEXT), ('trg', TEXT)]

dialogue_data = torchtext.data.TabularDataset(
    path='pairs450.tsv', format='tsv',
    fields=fields)

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

 
train_data, test_data = dialogue_data.split(split_ratio=0.95)
train_data, valid_data = train_data.split(split_ratio=0.95)

TEXT.build_vocab(train_data, min_freq = 3)
print(f"Unique tokens in vocabulary: {len(TEXT.vocab)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 8

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device, sort = False)


INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(TEXT.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = net.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = net.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = net.Seq2Seq(enc, dec, device).to(device)
model.apply(net.init_weights)

optimizer = optim.Adam(model.parameters())

PAD_IDX = TEXT.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')
for epoch in tqdm(range(N_EPOCHS)):
    start_time = time.time()
    train_loss = net.train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = net.evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    name=str(epoch)+'_tut1-model.pt'
    torch.save(model.state_dict(), name)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')