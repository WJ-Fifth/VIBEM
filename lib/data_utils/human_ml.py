# -*- coding: utf-8 -*-
# By Jinwu Wang
import os
import joblib
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm

from lib.core.config import VIBE_DB_DIR

dict_keys = ['betas', 'dmpls', 'gender', 'mocap_framerate', 'poses', 'trans']

# extract SMPL joints from SMPL-H model
joints_to_use = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 37
])
joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)

all_sequences = [
    'ACCAD',
    'BioMotionLab_NTroje',
    'CMU',
    'EKUT',
    'Eyes_Japan_Dataset',
    'HumanEva',
    'KIT',
    'MPI_HDM05',
    'MPI_Limits',
    'MPI_mosh',
    'SFU',
    'SSM_synced',
    'TCD_handMocap',
    'TotalCapture',
    'Transitions_mocap',
]


def read_data(folder, sequences):

    if sequences == 'all':
        sequences = all_sequences

    db = {
        'theta': [],
        'vid_name': [],
    }

    for seq_name in sequences:
        print(f'Reading {seq_name} sequence...')
        seq_folder = osp.join(folder, seq_name)

        thetas, vid_names = read_single_sequence(seq_folder, seq_name)
        seq_name_list = np.array([seq_name] * thetas.shape[0])
        print(seq_name, 'number of videos', thetas.shape[0])
        db['theta'].append(thetas)
        db['vid_name'].append(vid_names)

    db['theta'] = np.concatenate(db['theta'], axis=0)
    db['vid_name'] = np.concatenate(db['vid_name'], axis=0)

    return db


def read_single_sequence(folder, seq_name, fps=25):
    subjects = os.listdir(folder)

    thetas = []
    vid_names = []

    for subject in tqdm(subjects):
        actions = [x for x in os.listdir(osp.join(folder, subject)) if x.endswith('.npz')]

        for action in actions:
            fname = osp.join(folder, subject, action)

            if fname.endswith('shape.npz'):
                continue

            data = np.load(fname)

            mocap_framerate = int(data['mocap_framerate'])
            sampling_freq = mocap_framerate // fps
            pose = data['poses'][0::sampling_freq, joints_to_use]

            if pose.shape[0] < 60:
                continue

            shape = np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)
            theta = np.concatenate([pose, shape], axis=1)
            vid_name = np.array([f'{seq_name}_{subject}_{action[:-4]}'] * pose.shape[0])

            vid_names.append(vid_name)
            thetas.append(theta)

    return np.concatenate(thetas, axis=0), np.concatenate(vid_names, axis=0)


import shutil, sys
from torch.utils.data import Dataset
import glob



def dataset_exists(dataset_dir, split_names=None):
    '''
    This function checks whether a valid SuperCap dataset directory exists at a location
    Parameters
    ----------
    dataset_dir
    Returns
    -------
    '''
    if dataset_dir is None: return False
    if split_names is None:
        split_names = ['train', 'vald', 'test']
    import os

    import numpy as np

    done = []
    for split_name in split_names:
        for k in ['root_orient', 'pose_body']:#, 'betas', 'trans', 'joints']:
            outfname = os.path.join(dataset_dir, split_name, '%s.pt' % k)
            done.append(os.path.exists(outfname))
    return np.all(done)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/human_ml3d')
    args = parser.parse_args()

    db = read_data(args.dir, sequences=all_sequences)
    db_file = osp.join(VIBE_DB_DIR, 'human_ml3d_db.pt')
    print(f'Saving HumanML3D dataset to {db_file}')
    joblib.dump(db, db_file)
