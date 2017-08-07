"""
A classification model based on recurrent neural network with attention
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from fit_module import FitModule


class AttentionClassifier(FitModule):

    def __init__(self, batch_size, num_tokens, embed_size, gru_hidden, n_classes, bidirectional=True):

        super(AttentionClassifier, self).__init__()
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.gru_hidden = gru_hidden
        self.bidirectional = bidirectional
        self.n_classes = n_classes

        self.lookup = nn.Embedding(num_tokens, embed_size)
        self.gru = nn.GRU(embed_size, gru_hidden, bidirectional=True)
        self.weight_attention = nn.Parameter(torch.Tensor(2 * gru_hidden, 2 * gru_hidden))
        self.bias_attention = nn.Parameter(torch.Tensor(2 * gru_hidden, 1))
        self.weight_projection = nn.Parameter(torch.Tensor(2 * gru_hidden, 1))
        self.attention_softmax = nn.Softmax()
        self.final_softmax = nn.Linear(2 * gru_hidden, n_classes)
        self.weight_attention.data.uniform_(-0.1, 0.1)
        self.weight_projection.data.uniform_(-0.1, 0.1)
        self.bias_attention.data.uniform_(-0.1, 0.1)

    def batch_matmul_bias(self, seq, weight, bias, nonlinearity=''):
        s = None
        bias_dim = bias.size()
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
            if nonlinearity == 'tanh':
                _s_bias = torch.tanh(_s_bias)
            _s_bias = _s_bias.unsqueeze(0)
            if s is None:
                s = _s_bias
            else:
                s = torch.cat((s, _s_bias), 0)
        return s.squeeze()

    def batch_matmul(self, seq, weight, nonlinearity=''):
        s = None
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            if nonlinearity == 'tanh':
                _s = torch.tanh(_s)
            _s = _s.unsqueeze(0)
            if s is None:
                s = _s
            else:
                s = torch.cat((s, _s), 0)
        return s.squeeze()

    def attention_mul(self, rnn_outputs, att_weights):
        attn_vectors = None
        for i in range(rnn_outputs.size(0)):
            h_i = rnn_outputs[i]
            a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
            h_i = a_i * h_i
            h_i = h_i.unsqueeze(0)
            if attn_vectors is None:
                attn_vectors = h_i
            else:
                attn_vectors = torch.cat((attn_vectors, h_i), 0)
        return torch.sum(attn_vectors, 0)

    def forward(self, padded_sequence, initial_state):
        print padded_sequence
        embedded = self.lookup(padded_sequence)
        rnn_output, _ = self.gru(embedded.transpose(0,1), initial_state)
        attention_squish = self.batch_matmul_bias(rnn_output, self.weight_attention,
                                                  self.bias_attention, nonlinearity='tanh')
        attention = self.batch_matmul(attention_squish, self.weight_projection)
        attention_norm = self.attention_softmax(attention.transpose(1, 0))
        attention_vector = self.attention_mul(rnn_output, attention_norm.transpose(1, 0))
        linear_map = self.final_softmax(attention_vector)
        return linear_map

    def init_hidden(self):
        return Variable(torch.zeros(2, self.batch_size, self.gru_hidden))

