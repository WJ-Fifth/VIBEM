# -*- coding: utf-8 -*-
# By Mengfan Yan (u7375900)
# The code refers to the vibe model implemented by VIBE

import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from lib.core.config import VIBE_DATA_DIR
from lib.models.spin import Regressor, hmr


# Temporal Encoder

class TemporalEncoder(nn.Module):
    """
    recurrent architecture of the time series
    fed T frames to CNN, output vector: f_i of R^(2048)
    sent to GRU get the latent feature vector g_i
    sent g_i to regressor (SMPL)

    Parameters:
        num_layers: the number of the recurrent layers in GRU
        hidden_size: the number of hidden states in GRU (output size: hidden_size * D)
        linear: fed with one direction of GRU
        bidirectional: fed with double direction GRU
        residual:
    """

    def __init__(self, num_layers=1, hidden_size=2048, add_linear=False, bidirectional=False, residual=True):
        super(TemporalEncoder, self).__init__()
        # define with nn.LSTM

        self.lstm = nn.LSTM(input_size=2048,
                            hidden_size=hidden_size,
                            bidirectional=bidirectional,
                            num_layers=num_layers)

        # set one direction or two
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(2 * hidden_size, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
            # print("Test!!!=", self.linear)
        else:
            self.linear = None
        # set residual
        # print("hidden_size=", self.linear)
        self.residual = residual

    def forward(self, x):
        # number, time frames, feature
        n, t, f = x.shape
        x = x.permute(1, 0, 2)  # NTF -> TNF

        # LSTM
        lstm_output, lstm_state = self.lstm(x)


        ### LSTM
        if self.linear:
            # active layer, (seq,batchsize,hidden)
            lstm_output = F.relu(lstm_output)
            lstm_output = self.linear(lstm_output.view(-1, lstm_output.size(-1)))
            # set the default output formate
            lstm_output = lstm_output.view(t, n, f)
        # use residual
        if self.residual and lstm_output.shape[-1] == 2048:
            # print(lstm_output.shape)
            # print(x.shape)
            lstm_output = lstm_output + x
        # return to the input formate, (batchsize, sequences, inputsize)
        lstm_output = lstm_output.permute(1, 0, 2)
        return lstm_output


class VIBE_LSTM(nn.Module):
    """
    structure combine TE with Regressor
    TemporalEncoder + Regressor

    Parameters:
        sequences: length of time sequences
        batch: batch size
        num_layers: the number of the recurrent layers in GRU
        hidden_size: the number of hidden states in GRU (output size: hidden_size * D)
        linear: fed with one direction of GRU
        bidirectional: fed with double direction GRU
        residual:
        pre: pretrained model
    """

    def __init__(self, sequences, batch=64, num_layers=1, hidden_size=2048, linear=False, bidirectional=False,
                 residual=True,
                 pre=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar')):
        super(VIBE_LSTM, self).__init__()

        self.sequence = sequences
        self.batch = batch
        # set TemporalEncoder to encoder
        self.encoder = TemporalEncoder(num_layers=num_layers,
                                       hidden_size=hidden_size,
                                       add_linear=linear,
                                       bidirectional=bidirectional,
                                       residual=residual)
        # feed in regressor
        self.regressor = Regressor()
        # load pretrained model
        if pre and os.path.isfile(pre):
            pretrained_dict = torch.load(pre)['model']
            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print("pretrained model with " + str(pre))

    def forward(self, input, J_regressor=None):
        # input size (batchsize,sequence,feature)
        batch = input.shape[0]
        sequence = input.shape[1]

        lstm_feature = self.encoder(input)
        lstm_feature = lstm_feature.reshape(-1, lstm_feature.size(-1))

        smpl_output = self.regressor(lstm_feature, J_regressor=J_regressor)
        for k in smpl_output:
            k['theta'] = k['theta'].reshape(batch, sequence, -1)
            k['verts'] = k['verts'].reshape(batch, sequence, -1, 3)
            k['kp_2d'] = k['kp_2d'].reshape(batch, sequence, -1, 2)
            k['kp_3d'] = k['kp_3d'].reshape(batch, sequence, -1, 3)
            # (batch_size, 24, 3, 3)
            k['rotmat'] = k['rotmat'].reshape(batch, sequence, -1, 3, 3)
        return smpl_output


class VIBE_LSTM_Demo(nn.Module):
    """
    structure combine TE with Regressor
    TemporalEncoder + Regressor

    Parameters:
        sequences: length of time sequences
        batch: batch size
        num_layers: the number of the recurrent layers in GRU
        hidden_size: the number of hidden states in GRU (output size: hidden_size * D)
        linear: fed with one direction of GRU
        bidirectional: fed with double direction GRU
        residual:
        pre: pretrained model
        hmr: pretained model to test
    """

    def __init__(self, sequences, batch=64,
                 num_layers=1,
                 hidden_size=2048,
                 linear=False,
                 bidirectional=False,
                 residual=True,
                 pre=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar')):
        super(VIBE_LSTM_Demo, self).__init__()

        self.sequence = sequences
        self.batch = batch
        print(linear)
        # set TemporalEncoder to encoder
        self.encoder = TemporalEncoder(num_layers=num_layers,
                                       hidden_size=hidden_size,
                                       add_linear=linear,
                                       bidirectional=bidirectional,
                                       residual=residual)
        # pretrained model to test
        self.hmr = hmr()
        checkpoint = torch.load(pre)
        self.hmr.load_state_dict(checkpoint['model'], strict=False)

        # feed in regressor
        self.regressor = Regressor()
        # load pretrained model
        if pre and os.path.isfile(pre):
            pretrained_dict = torch.load(pre)['model']
            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print("pretrained model with " + str(pre))

    def forward(self, input_size, regress=None):
        # input size (batchsize,sequence,feature)
        batch = input_size.shape[0]
        sequence = input_size.shape[1]
        n, h, w = input_size.shape[2:]
        # hmr with feature extract
        hmr_output = self.hmr.feature_extractor(input_size.reshape(-1, n, h, w))
        # TemporalEncoder
        lstm_output = self.encoder(hmr_output.reshape(batch, sequence, -1))
        # regressor
        lstm_output = lstm_output.reshape(-1, lstm_output.size(-1))
        reg_output = self.regressor(lstm_output, J_regressor=regress)
        for k in reg_output:
            k['theta'] = k['theta'].reshape(batch, sequence, -1)
            k['verts'] = k['verts'].reshape(batch, sequence, -1, 3)
            k['kp_2d'] = k['kp_2d'].reshape(batch, sequence, -1, 2)
            k['kp_3d'] = k['kp_3d'].reshape(batch, sequence, -1, 3)
            # (batch_size, 24, 3, 3)
            k['rotmat'] = k['rotmat'].reshape(batch, sequence, -1, 3, 3)
        return reg_output


## Test with VIBEM model
if __name__ == "__main__":
    from torchsummary import summary

    input_size = (16, 2048)

    model = VIBE_LSTM(sequences=16, batch=64)
    print(model)
    summary(model, input_size=[2048], device="cpu")
