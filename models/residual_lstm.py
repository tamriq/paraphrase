import random

import torch
import torch.nn as nn
from torch.autograd import Variable


class ResidualEncoder(nn.Module):
    """
    LSTM encoder with the residual connection.
    """

    def __init__(self, input_dim, hidden_enc_dim, hidden_dec_dim):
        super().__init__()
        #
        self.emb = nn.Embedding(input_dim, hidden_enc_dim)
        #
        self.first_lstm = nn.LSTM(hidden_enc_dim, hidden_enc_dim, num_layers=2,
                                  bidirectional=False, batch_first=True, dropout=0.3)
        #
        self.second_lstm = nn.LSTM(hidden_enc_dim, hidden_enc_dim, num_layers=2,
                                   bidirectional=False, batch_first=True, dropout=0.3)
        #
        self.third_lstm = nn.LSTM(hidden_enc_dim, hidden_dec_dim, num_layers=2,
                                  bidirectional=False, batch_first=True, dropout=0.3)

    def forward(self, x):
        x_emb = self.emb(x)
        outputs_1, (hidden, cell) = self.first_lstm(x_emb)
        #
        outputs_1 = torch.add(x_emb, outputs_1)
        #
        outputs_2, (hidden, cell) = self.second_lstm(outputs_1, (hidden, cell))
        #
        outputs_2 = torch.add(outputs_2, outputs_1)
        #
        outputs_3, (hidden, cell) = self.third_lstm(outputs_2, (hidden, cell))
        #
        outputs_3 = torch.add(outputs_3, outputs_2)
        return outputs_3, (hidden, cell)


class ResidualDecoder(nn.Module):
    """
    LSTM decoder with the residual connection.
    """

    def __init__(self, vocab_size, hidden_dec_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dec_dim)
        # Maybe here as the first element hidden_enc_dim?
        self.first_lstm = nn.LSTM(hidden_dec_dim, hidden_dec_dim, num_layers=2,
                                  bidirectional=False, batch_first=True, dropout=0.3)
        self.second_lstm = nn.LSTM(hidden_dec_dim, hidden_dec_dim, num_layers=2,
                                   bidirectional=False, batch_first=True, dropout=0.3)
        self.third_lstm = nn.LSTM(hidden_dec_dim, hidden_dec_dim, num_layers=2,
                                  bidirectional=False, batch_first=True, dropout=0.3)
        self.out = nn.Linear(hidden_dec_dim, vocab_size)

    def lstm(self, embedded_dec, enc_memory):
        outputs_1, (hidden, cell) = self.first_lstm(embedded_dec, enc_memory)
        outputs_1 = torch.add(embedded_dec, outputs_1)
        outputs_2, (hidden, cell) = self.second_lstm(outputs_1, (hidden, cell))
        outputs_2 = torch.add(outputs_2, outputs_1)
        outputs_3, (hidden, cell) = self.third_lstm(outputs_2, (hidden, cell))
        return outputs_2, (hidden, cell)

    def forward(self, dec_seq, enc_memory):
        embedded_dec = self.emb(dec_seq)
        outputs, mem = self.lstm(embedded_dec, enc_memory)
        outputs = self.out(outputs)
        return outputs, mem


class Seq2SeqResidual(nn.Module):
    """

    """

    def __init__(self, encoder, decoder, batch_size, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.vocab_size = vocab_size

    def forward(self, x, y):
        #
        enc_outputs, (hidden, cell) = self.encoder(x)
        max_len = y.size(1)
        outputs = torch.zeros(self.batch_size, max_len, self.vocab_size)
        teacher_forcing_ratio = 0.4
        #
        inp = y[:, 0].unsqueeze(1)
        #
        for t in range(1, max_len):
            #
            output, (hidden, cell) = self.decoder(inp, (hidden, cell))
            #
            outputs[:, t] = output.squeeze(1)
            #
            use_teacher_force = random.random() < teacher_forcing_ratio
            #
            top1 = output.max(2)[1]
            inp = y[:, t].unsqueeze(1) if use_teacher_force else top1
        return outputs


class ResidualDecoderAttention(nn.Module):
    """
    LSTM decoder with residual connection and attention.
    """

    def __init__(self, input_dim, hidden_enc_dim, hidden_dec_dim):
        super().__init__()
        self.emb = nn.Embedding(input_dim, hidden_dec_dim)
        self.first_lstm = nn.LSTM(hidden_dec_dim + hidden_dec_dim, hidden_dec_dim, num_layers=2,
                                  bidirectional=True, batch_first=True)
        self.second_lstm = nn.LSTM(hidden_dec_dim, hidden_dec_dim, num_layers=2,
                                   bidirectional=True, batch_first=True)
        self.third_lstm = nn.LSTM(hidden_dec_dim, hidden_dec_dim, num_layers=2,
                                  bidirectional=True, batch_first=True)
        # fc
        self.out = nn.Linear(hidden_dec_dim, input_dim)
        #
        self.W1 = nn.Linear(hidden_enc_dim, hidden_dec_dim)
        self.W2 = nn.Linear(hidden_enc_dim, hidden_dec_dim)
        self.V = nn.Linear(hidden_enc_dim, 1)

    def lstm(self, x):
        outputs_1, (hidden, cell) = self.first_lstm(x)
        outputs_2, (hidden, cell) = self.second_lstm(outputs_1, (hidden, cell))
        # Apply residual connection after the second layer of LSTM.
        outputs_2 = torch.add(outputs_2, outputs_1)
        outputs_3, (hidden, cell) = self.third_lstm(outputs_2, (hidden, cell))
        # Apply residual connection after the third layer of LSTM.
        outputs_3 = torch.add(outputs_3, outputs_2)
        return outputs_3, (hidden, cell)

    def forward(self, dec_seq, hidden, enc_output):
        # Get the last hidden layer from the encoder.
        hidden_with_time_axis = hidden[0][1].unsqueeze(1)
        # Get the attention score.
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        # Get the attention weights.
        attention_weights = torch.softmax(self.V(score), dim=1)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        x = self.emb(dec_seq)[0]
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        output, state = self.lstm(x)
        output = output.view(-1, output.size(2))
        out = self.out(output)
        return out, state, attention_weights


class Seq2seqAttention(nn.Module):
    """
    """

    def __init__(self, encoder, decoder, vocab_size, batch):
        super().__init__()
        self.residual_encoder = encoder
        self.residual_decoder = decoder
        self.vocab_size = vocab_size
        self.batch_size = batch

    def forward(self, x, y):
        # Prepare the storage for the outputs.
        outputs = Variable(torch.zeros(self.batch_size, y.size(1), self.vocab_size))
        # Process the source sequences with the encoder.
        encoder_output, hidden = self.residual_encoder(x)
        teacher_forcing_ratio = 0.5
        output = y[:, 0].unsqueeze(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output = output.to(device)
        #
        for t in range(1, y.size(1)):
            output, hidden, attn_weights = self.residual_decoder(output, hidden, encoder_output)
            outputs[:, t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = y[:, t].unsqueeze(1) if is_teacher else top1.unsqueeze(1)
        return outputs
