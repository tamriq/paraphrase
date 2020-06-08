class SequenceBucketingData(torch.utils.data.Dataset):
    
    def __init__(self, data, max_len, pad_index, eos_index, bos_index):
        
        self.data = data
        
        self.max_len = max_len
        
        self.pad_index = pad_index
        self.eos_index = eos_index
        self.bos_index = bos_index
        
    def __len__(self):
        
        return len(self.data)
    
    def prepare_sample(self, sequence, max_len_dec):
        
        enc = sequence[0][:self.max_len]
        dec = sequence[1][:max_len_dec]
        
        enc = [self.bos_index] + enc + [self.eos_index]
        dec = [self.bos_index] + dec + [self.eos_index]
                
        pads_enc = [self.pad_index] * (self.max_len+2 - len(enc))
        pads_dec = [self.pad_index] * (max_len_dec+2 - len(dec))
        enc += pads_enc
        dec += pads_dec
        
        return enc, dec
    
    def __getitem__(self, index):
        
        batch = self.data[index]

        max_len_dec = min([self.max_len, max([len(sample[1]) for sample in batch])])
        
        batch_x = []
        batch_y = []
        
        for sample in batch:
            x, y = self.prepare_sample(sample, max_len_dec)
            batch_x.append(x)
            batch_y.append(y)
        
        batch_x = torch.tensor(batch_x).long()
        batch_y = torch.tensor(batch_y).long()
        
        return batch_x, batch_y