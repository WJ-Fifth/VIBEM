# -*- coding: utf-8 -*-
# By Jinwu Wang (u7354172)

import torch
import joblib
import numpy as np
import os.path as osp
from torch.utils.data import Dataset

from lib.core.config import VIBE_DB_DIR
from lib.data_utils.img_utils import split_into_chunks

class HumanML3D(Dataset):
    def __init__(self, seqlen):
        self.seqlen = seqlen

        self.stride = seqlen

        self.db = self.load_db()
        self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)
        del self.db['vid_name']
        print(f'Human_ML3D dataset number of videos: {len(self.vid_indices)}')

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db_file = osp.join(VIBE_DB_DIR, 'human_ml3d_db.pt')
        db = joblib.load(db_file)
        return db

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]
        thetas = self.db['theta'][start_index:end_index+1]

        cam = np.array([1., 0., 0.])[None, ...]
        cam = np.repeat(cam, thetas.shape[0], axis=0)
        theta = np.concatenate([cam, thetas], axis=-1)

        target = {
            'theta': torch.from_numpy(theta).float(),  # cam, pose and shape
        }
        return target



