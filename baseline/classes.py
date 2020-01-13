class WordData(torch.utils.data.Dataset):
    def __init__(self, context_list, questions_list, context_len, questions_len, pad_index, eos_index):
        self.context_list = context_list
        self.questions_list = questions_list
        
        self.context_len = context_len
        self.questions_len = questions_len
        
        self.pad_index = pad_index
        self.eos_index = eos_index
        
    def __len__(self):
        return len(self.context_list)
    
    def __getitem__(self, index):
        
        context = self.context_list[index][:self.context_len]
        pads_ctx = [self.pad_index] * (self.context_len - len(context))
#         print(len(pads_ctx))
        context = torch.tensor(context + pads_ctx).long()
        
        question = self.questions_list[index][:self.questions_len]
        pads_quest = [self.pad_index] * (self.questions_len - len(question))
        question = torch.tensor(question + pads_quest).long()
        
        return context, question