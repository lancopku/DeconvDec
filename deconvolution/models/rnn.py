import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models


class LayerNorm(nn.Module):
    """
        Layer Normalization class
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class rnn_encoder_layer(nn.Module):

    def __init__(self, config):
        super(rnn_encoder_layer, self).__init__()

        self.ln = LayerNorm(config.hidden_size)
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=1, bidirectional=config.bidirectional)
        self.hidden_size = config.hidden_size
        self.config = config
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        input_norm = self.ln(inputs)
        contexts, state = self.rnn(input_norm)
        if self.config.bidirectional:
            contexts = contexts[:, :, :self.hidden_size] + contexts[:, :, self.hidden_size:]
        out = self.dropout(contexts + inputs)
        return out, state


class rnn_encoder(nn.Module):

    def __init__(self, config, embedding=None):
        super(rnn_encoder, self).__init__()

        self.embedding = embedding if embedding is not None else nn.Embedding(config.src_vocab_size, config.emb_size)
        self.hidden_size = config.hidden_size
        self.config = config
        if config.attention == 'None':
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong':
            self.attention = models.luong_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong_gate':
            self.attention = models.luong_gate_attention(config.hidden_size, config.emb_size)
        if config.cell == 'gru':
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                              num_layers=config.enc_num_layers, dropout=config.dropout,
                              bidirectional=config.bidirectional)
        else:
            # self.rnn_enc = nn.ModuleList([rnn_encoder_layer(config)
            #                           for _ in range(config.enc_num_layers)])
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                               num_layers=config.enc_num_layers, dropout=config.dropout,
                               bidirectional=config.bidirectional)
        self.ln = LayerNorm(config.hidden_size)

    def forward(self, inputs, lengths):
        embs = pack(self.embedding(inputs), lengths)
        # out = unpack(embs)[0]
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        # h_list, c_list = [], []
        # for i in range(self.config.enc_num_layers):
        #     out, state = self.rnn_enc[i](out)
        #     h, c = state[0], state[1]
        #     h_list.append(h[0])
        #     h_list.append(h[1])
        #     c_list.append(c[0])
        #     c_list.append(c[1])
        # outputs = self.ln(out)
        # h = torch.stack(h_list)
        # c = torch.stack(c_list)
        # state = (h[::2], c[::2])
        if self.config.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        if self.config.cell == 'gru':
            state = state[:self.config.dec_num_layers]
        else:
            state = (state[0][::2], state[1][::2])

        return outputs, state



class rnn_decoder(nn.Module):

    def __init__(self, config, embedding=None, use_attention=True):
        super(rnn_decoder, self).__init__()
        self.embedding = embedding if embedding is not None else nn.Embedding(config.tgt_vocab_size, config.emb_size)

        input_size = config.emb_size

        if config.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.dec_num_layers, dropout=config.dropout)
        else:
            self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                   num_layers=config.dec_num_layers, dropout=config.dropout)

        self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)

        if not use_attention or config.attention == 'None':
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong':
            self.attention = models.luong_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong_gate':
            self.attention = models.luong_gate_attention(config.hidden_size, config.emb_size)

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, input, state, convs):
        embs = self.embedding(input)
        output, state = self.rnn(embs, state)
        if self.attention is not None:
            if self.config.attention == 'luong_gate':
                output, attn_weights, weights_conv = self.attention(output, convs)
            else:
                output, attn_weights = self.attention(output, embs)
        else:
            attn_weights = None
            weights_conv = None
        # output = self.dropout(output)
        output = self.compute_score(output)

        return output, state, attn_weights, weights_conv

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores


class conv_decoder(nn.Module):
    def __init__(self, config, embedding=None, use_attention =True):
        super(conv_decoder, self).__init__()

        self.embedding = embedding if embedding is not None else nn.Embedding(config.tgt_vocab_size, config.emb_size)
        if not use_attention or config.attention == 'None':
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong':
            self.attention = models.luong_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong_gate':
            self.attention_1 = models.luong_gate_attention(config.hidden_size, config.emb_size, selfatt=True)
            self.attention_2 = models.luong_gate_attention(config.hidden_size, config.emb_size, selfatt=True)
            self.attention_3 = models.luong_gate_attention(config.hidden_size, config.emb_size, selfatt=True)

        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = config.hidden_size
        self.config = config

        self.deconv_1 = nn.Sequential(nn.ConvTranspose1d(config.hidden_size, config.hidden_size, kernel_size=2, stride=1, padding=0),
                                      nn.ReLU(), nn.Dropout(config.dropout))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2, padding=0),
                                      nn.ReLU(), nn.Dropout(config.dropout))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose1d(config.hidden_size, config.hidden_size, kernel_size=4, stride=3, padding=0),
                                      nn.ReLU(), nn.Dropout(config.dropout))

        # self.ln_1, self.ln_2, self.ln_3, self.ln_4 = LayerNorm(config.hidden_size), LayerNorm(config.hidden_size), LayerNorm(config.hidden_size), LayerNorm(config.hidden_size)
        # self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)
        # self.linear = lambda x: torch.matmul(x, Variable(self.embedding.weight.t().data)
        self.linear = lambda x: torch.matmul(x, torch.tensor(self.embedding.weight.t(), requires_grad=False))

    def forward(self, state):
        tmp = state[0].transpose(0,1).transpose(1,2) #N*L*C
        # tmp = self.ln_1(inp).transpose(1,2) #N*C*L
        tmp = self.deconv_1(tmp) #N*C*L
        tmp = self.deconv_2(tmp)
        tmp = self.deconv_3(tmp)
        output = tmp.transpose(1,2)
        # inp = tmp.transpose(1,2) #N*L*C
        # # inp = self.attention_1(tmp, inp) #N*L*C
        # tmp = self.ln_2(inp).transpose(1,2) #N*C*L
        # tmp = self.deconv_2(tmp) #N*C*L
        # # inp = self.attention_2(tmp, inp) #N*L*C
        # inp = tmp.transpose(1, 2)  # N*L*C
        # tmp = self.ln_3(inp).transpose(1,2) #N*C*L
        # tmp = self.deconv_3(tmp) #N*C*L
        # # inp = self.attention_3(tmp, inp) #N*L*C
        # inp = tmp.transpose(1, 2)  # N*L*C
        # output = self.ln_4(inp) #N*L*C
        scores = self.compute_score(output)

        return output, scores

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        # self.ln = LayerNorm(hidden_size)

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            # input_norm = self.ln(input)
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        # output = self.ln(input)
        output = input
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return output, (h_1, c_1)


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()


        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1
