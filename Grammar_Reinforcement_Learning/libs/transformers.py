import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.nn import (TransformerEncoder,TransformerEncoderLayer,
                      TransformerDecoder,TransformerDecoderLayer,
                      Embedding)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len,1, d_model)
        pe[:,0, 0::2] = torch.sin(position * div_term)
        pe[:,0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class grammar_encoder(nn.Module):
    def __init__(self,ntoken,d_model = 128,nhead = 4,d_hid = 1024, 
                 num_layers = 5, dropout = .1):
        super(grammar_encoder, self).__init__()
        
        self.pos_encoder = PositionalEncoding(d_model,dropout)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                dim_feedforward = d_hid,dropout = dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        self.embedding = nn.Embedding(ntoken,d_model)
        self.d_model = d_model
        
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        
        
        
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        # print(src.shape)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output
    
class grammar_decoder(nn.Module):
    def __init__(self,ntoken,d_model = 128,nhead = 4,d_hid = 1024, 
                 num_layers = 5, dropout = .1):
        super(grammar_decoder, self).__init__()
        
        self.pos_encoder = PositionalEncoding(d_model,dropout)
        
        decoder_layer = TransformerDecoderLayer(d_model = d_model, nhead = nhead,
                                                dim_feedforward = d_hid,dropout = dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)
        self.embedding = nn.Embedding(ntoken,d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
        
        
    def forward(self, memory: Tensor, target: Tensor, src_mask=None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        
        
        target = self.embedding(target) * math.sqrt(self.d_model)
        target = self.pos_encoder(target)
        if src_mask == None:
            output = self.transformer_decoder(target,memory)
        else:
            output =  self.transformer_decoder(target,memory,tgt_key_padding_mask = src_mask)
        # output = self.linear(output)
        return output
