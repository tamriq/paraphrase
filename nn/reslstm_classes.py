class ResidualEncoder(nn.Module):
    def __init__(self, input_dim, hidden_enc_dim, hidden_dec_dim):
        super().__init__()
        # сделать как в статье bi-directionsl подушку первым слоем?

        self.emb = nn.Embedding(input_dim, hidden_enc_dim)
        self.first_two = nn.LSTM(hidden_enc_dim, hidden_enc_dim, num_layers=2, bidirectional = False, batch_first = True, dropout=0.3)
        self.sec_two = nn.LSTM(hidden_enc_dim, hidden_enc_dim, num_layers=2, bidirectional = False, batch_first = True, dropout=0.3)
        self.third_two = nn.LSTM(hidden_enc_dim, hidden_enc_dim, num_layers=2, bidirectional = False, batch_first = True, dropout=0.3)
        
        
    def forward(self, x):
        
        x_emb = self.emb(x)
                
        outputs_1, (hidden, cell) = self.first_two(x_emb)
        outputs_1 = torch.add(x_emb, outputs_1)
        
        outputs_2, (hidden, cell) = self.sec_two(outputs_1, (hidden, cell))
        outputs_2 = torch.add(outputs_2, outputs_1)
        
        outputs_3, (hidden, cell) = self.third_two(outputs_2, (hidden, cell))
        outputs_3=torch.add(outputs_3, outputs_2)

        return outputs_3, (hidden, cell)


class ResidualDecoder(nn.Module):
    def __init__(self, input_dim, hidden_enc_dim, hidden_dec_dim):
        super().__init__()
        
        self.emb = nn.Embedding(input_dim, hidden_dec_dim)
        
        self.first_two = nn.LSTM(hidden_dec_dim, hidden_dec_dim, num_layers=2, bidirectional = False, batch_first = True, dropout=0.3)
        self.sec_two = nn.LSTM(hidden_dec_dim, hidden_dec_dim, num_layers=2, bidirectional = False, batch_first = True, dropout=0.3)
        self.third_two = nn.LSTM(hidden_dec_dim, hidden_dec_dim, num_layers=2, bidirectional = False, batch_first = True, dropout=0.3)
        self.out = nn.Linear(hidden_dec_dim, input_dim)
        
         
    def lstm(self, dec_seq, enc_memory):
        
        embedded = self.emb(dec_seq)
        
        outputs_1, (hidden, cell) = self.first_two(embedded, enc_memory) # enc memory
        
        outputs_1 = torch.add(embedded, outputs_1)
        
        outputs_2, (hidden, cell) = self.sec_two(outputs_1, (hidden, cell))
        outputs_2 = torch.add(outputs_2, outputs_1)
        
        outputs_3, (hidden, cell) = self.third_two(outputs_2, (hidden, cell))
        
        return outputs_2, (hidden, cell)
        
    def forward(self, dec_seq, enc_memory):
        
        
        outputs, mem = self.lstm(dec_seq, enc_memory)
        
        outputs = self.out(outputs)
        
        return outputs, mem
        


class Seq2SeqResidual(nn.Module):
    def __init__(self, encoder, decoder, batch_size, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    
    
    def forward(self, x, y):
        enc_outputs, (hidden, cell) = self.encoder(x)
        
        max_len = y.size(1)
        batch_size = y.size(0)
        
        outputs = torch.zeros(batch_size, max_len, vocab_size)
        teacher_forcing_ratio=0.4
        inp = y[:, 0].unsqueeze(1)
     
        for t in range(1, max_len):
            
            output, (hidden, cell) = self.decoder(inp, (hidden, cell))
            
            outputs[:,t] = output.squeeze(1)
            
            use_teacher_force = random.random() < teacher_forcing_ratio
            
            
            top1 = output.max(2)[1]
            
            
            inp = y[:, t].unsqueeze(1) if use_teacher_force else top1
   
                    
        return outputs



class ResidualDecoderAttention(nn.Module):
    def __init__(self, input_dim, hidden_enc_dim, hidden_dec_dim, embedder):
        super().__init__()
        
        #self.emb = nn.Embedding(input_dim, hidden_dec_dim)
        self.emb = embedder
        self.first_two = nn.LSTM(768+hidden_dec_dim, hidden_dec_dim, num_layers=2, bidirectional = True, batch_first = True)
        self.sec_two = nn.LSTM(hidden_dec_dim, hidden_dec_dim, num_layers=2, bidirectional = True, batch_first = True)
        self.third_two = nn.LSTM(hidden_dec_dim, hidden_dec_dim, num_layers=2, bidirectional = True, batch_first = True)
        
        self.out = nn.Linear(hidden_dec_dim, input_dim) #fc
        
        # attention
        self.W1 = nn.Linear(hidden_enc_dim, hidden_dec_dim)
        self.W2 = nn.Linear(hidden_enc_dim, hidden_dec_dim)
        self.V = nn.Linear(hidden_enc_dim, 1)        
         
    def lstm(self, x):
        
        
        outputs_1, (hidden, cell) = self.first_two(x) # enc memory
            
        outputs_2, (hidden, cell) = self.sec_two(outputs_1, (hidden, cell))
        outputs_2 = torch.add(outputs_2, outputs_1)
        
        outputs_3, (hidden, cell) = self.third_two(outputs_2, (hidden, cell))
        outputs_3 = torch.add(outputs_3, outputs_2)
        
        return outputs_3, (hidden, cell)
        
    def forward(self, dec_seq, hidden, enc_output):
        
        
        
        hidden_with_time_axis = hidden[0][1].unsqueeze(1) #последний слой
        

        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        
    
        attention_weights = torch.softmax(self.V(score), dim=1)
        
        context_vector = attention_weights * enc_output
        
        context_vector = torch.sum(context_vector, dim=1)
        
        
        x=self.emb(dec_seq)[0]
    
        x = torch.cat((context_vector.unsqueeze(1), x), -1)  
        
        
        output, state = self.lstm(x)
        
        
        
        output =  output.view(-1, output.size(2))


        out = self.out(output)
        
        
        return out, state, attention_weights





class Seq2seqAttention(nn.Module):
    
    def __init__(self, encoder, decoder, vocab_size, batch):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.batch_size = batch
        
        
    def forward(self, x,y):
        
        outputs = Variable(torch.zeros(self.batch_size, y.size(1), self.vocab_size))

        encoder_output, hidden = res_encoder(x)

        teacher_forcing_ratio = 0.5

        output = y[:,0].unsqueeze(1) # sos
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output=output.to(device)
        
        for t in range(1, y.size(1)):
    
            output, hidden, attn_weights = att_decoder(output, hidden, encoder_output)
            
            outputs[:,t] = output
    
            is_teacher = random.random() < teacher_forcing_ratio
    
            top1 = output.max(1)[1]
    
            output = y[:,t].unsqueeze(1) if is_teacher else top1.unsqueeze(1)
        
        return outputs


        