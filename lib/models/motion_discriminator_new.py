# -*- coding: utf-8 -*-
# By Mengfan Yan (u7375900)
# The code refers to the motion_discriminator.py implemented by VIBE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from lib.models.attention import SelfAttention


class MotionDiscriminator_geo(nn.Module):
    """
    Discriminator with GRU and self attention

    Parameter:
    rsize: the number of RNN layers in GRU
    insize: input size of Discriminator/GRU
    num_layers: the number of GRU layers
    outsize: output size with FC layer to combine self attention results
    pool: choose to introduce self attention
    use_spectral_norm:
    atsize: attention size
    atlayers: attention layers
    atdropout: dropout param of self attention
    lsize: linear size
    """

    def __init__(self,
                 rsize,
                 insize,
                 num_layers,
                 outsize=2,
                 pool="attention",
                 use_spectral_norm=False,
                 atsize=1024,
                 atlayers=1,
                 atdropout=0.5):
        super(MotionDiscriminator_geo, self).__init__()
        self.rsize = rsize
        self.insize = insize
        self.num_layers = num_layers
        self.pool = pool
        self.atsize = atsize
        self.atlayers = atlayers
        self.atdropout = atdropout
        # fed in GRU and check to concat(depend on the number of gru layers)
        # self.gru = nn.GRU(self.insize, self.rsize, num_layers=self.num_layers)

        # change GRU to LSTM
        self.lstm = nn.LSTM(self.insize, self.rsize, num_layers=self.num_layers)

        # only with attention
        lsize = self.rsize
        self.attention = SelfAttention(attention_size=self.atsize, layers=self.atlayers, dropout=self.atdropout)

        if use_spectral_norm:
            self.fc = spectral_norm(nn.Linear(lsize, outsize))
        else:
            self.fc = nn.Linear(lsize, outsize)

    def forward(self, input):
        """
        input: (batch size, sequence, input size)
        """
        batch, sequence, insize = input.shape

        # fed in LSTM (sequence, batch, input)
        lstm_output, lstm_state = self.lstm(input.permute(1, 0, 2))

        # introduce attention or just concat
        if self.pool == "attention":
            at_output, scores = self.attention(lstm_output.permute(1, 0, 2))
            output = self.fc(at_output)
        elif self.feature_pool == "concat":
            outputs = F.relu(lstm_output)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batch, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batch, -1)
            output = self.fc(torch.cat([avg_pool, max_pool], dim=1))
        else:
            output = self.fc(lstm_output[-1])
        return output
