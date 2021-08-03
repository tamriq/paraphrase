import torch
from torch import nn


class Encoder(nn.Module):
    """
    The whole Encoder part of the Transformer neural network, contains all encoder layers.
    """

    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()
        """

        :param input_dim:
        :param hid_dim:
        :param n_layers:
        :param n_heads:
        :param pf_dim:
        :param dropout:
        :param device:
        :param max_length:
        """
        self.device = device
        self.get_embedding = nn.Embedding(input_dim, hid_dim)
        # Get positional embedding.
        self.get_pos_embedding = nn.Embedding(max_length, hid_dim)
        # Combine all layers of the encoder into one pipeline.
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])
        #
        self.dropout = nn.Dropout(dropout)
        #
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        """

        :param src:
        :param src_mask:
        :return:
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]
        # Get the positional embedding of the sequence.
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # Get the final sequence embedding.
        src = (self.get_embedding(src) * self.scale) + self.get_pos_embedding(pos)
        # Apply dropout.
        src = self.dropout(src)
        # Iteratively step through all layers of the network.
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class EncoderLayer(nn.Module):
    """
    Computes one layer of the transformer Encoder.
    """

    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()
        """
        Initialize the EncoderLayer class parameters.

        :param hid_dim: the size of hidden dimension
        :param n_heads: the number of heads of multi-head attention layer
        :param pf_dim: the size of positionwise feedforward layer (should be bigger than hid_dim)
        :param dropout: tha amount of dropout (between 0 and 1)
        :param device: the specified device to compute on
        """
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """
        Process the sequence with the encoder layer.

        :param src:
        :param src_mask:
        :return:
        """
        # Get the self-attention weights for the sequence.
        _src, _ = self.self_attention(src, src, src, src_mask)
        # Apply layer normalization.
        src = self.layer_norm(src + self.dropout(_src))
        #
        _src = self.positionwise_feedforward(src)
        #
        src = self.layer_norm(src + self.dropout(_src))
        return src


class MultiHeadAttentionLayer(nn.Module):
    """

    """

    def __init__(self, hid_dim, n_heads, dropout, device):
        """

        :param hid_dim:
        :param n_heads:
        :param dropout:
        :param device:
        """
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        #
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        """

        :param query:
        :param key:
        :param value:
        :param mask:
        :return:
        """
        batch_size = query.shape[0]
        #
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        #
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        #
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            #
            energy = energy.masked_fill(mask == 0, -1e10)
        #
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        """

        :param hid_dim:
        :param pf_dim:
        :param dropout:
        """
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class Decoder(nn.Module):
    """

    """

    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()
        self.device = device
        #
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        #
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        #
        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """

        :param trg:
        :param enc_src:
        :param trg_mask:
        :param src_mask:
        :return:
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        #
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        #
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention


class DecoderLayer(nn.Module):
    """

    """

    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        """
        Initialize the DecoderLayer class parameters.

        :param hid_dim: the size of hidden dimension
        :param n_heads: the number of heads of multi-head attention layer
        :param pf_dim: the size of positionwise feedforward layer (should be bigger than hid_dim)
        :param dropout: tha amount of dropout (between 0 and 1)
        :param device: the specified device to compute on
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """

        :param trg:
        :param enc_src:
        :param trg_mask:
        :param src_mask:
        :return:
        """
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.layer_norm(trg + self.dropout(_trg))
        return trg, attention


class Seq2Seq(nn.Module):
    """
    Encapsulates the encoder and decoder classes.
    """

    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        """
        Initialize Seq2Seq class.

        :param encoder: initialized Encoder class
        :param decoder: initialized Decoder class
        :param src_pad_idx: the index of the padding token for the source sequence
        :param trg_pad_idx: the index of the padding token for the target sequence
        :param device: the specified device to compute on
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def mask_src(self, src):
        """
        Apply the mask to the source sequence.

        :param src: source sequence
        :return: masked source sequence
        """
        # Mask the tokens which are not equal to the <pad> token.
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def mask_trg(self, trg):
        """
        Apply the mask to the target sequence.

        :param trg: target sequence
        :return: masked target sequence
        """
        # Mask the tokens which are not equal to the <pad> token.
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        # Create subsequent mask for the sequence.
        # This mask is a diagonal matrix which shows what each target token (row) is
        # allowed to look at (column). E.g the first target token has a mask of [1, 0, 0, 0, 0]
        # which means it can only look at the first target token.
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # Combine the masks.
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.mask_src(src)
        trg_mask = self.mask_trg(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention


def _initialize_weights_xavier(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)


def build_model(config):
    model_parts = {"enc": Encoder, "dec": Decoder}
    for name, part in model_parts.items():
        model_parts[name] = part(config["data"]["vocab_size"],
                                 config["model"]["enc_hid_dim"],
                                 config["model"]["n_enc_layers"],
                                 config["model"]["n_enc_heads"],
                                 config["model"]["enc_pf_dim"],
                                 config["model"]["enc_dropout"],
                                 config["device"])
    # Initialize the Transformer model and move it to the specified device.
    model = Seq2Seq(model_parts["enc"], model_parts["dec"], config["data"]["pad_idx"],
                         config["data"]["pad_idx"], config["device"]).to(config["device"])
    model.apply(_initialize_weights_xavier)
    return model