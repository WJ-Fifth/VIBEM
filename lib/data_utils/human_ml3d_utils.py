# -*- coding: utf-8 -*-
# By Ruijia Tan (u7383615)
# Implement data extraction and packaging of HumanML3D dataset

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


def read_data(folder, file_dir):
    """
    :param folder: Dataset storage location
    :param file_dir: File address, the file saves the data set information that needs to be extracted
    :return: Extracted and packed formatted dataset

    """
    sequences = osp.join(folder, file_dir)

    data_list = os.listdir(sequences)

    database = {
        'theta': [],
        'vid_name': [],
    }

    for seq_name in data_list:
        # Loop through all data in a folder
        print(f'Reading {seq_name} sequence...')
        seq_folder = osp.join(folder, seq_name)

        thetas, vid_names = read_single_sequence(seq_folder, seq_name)
        print(seq_name, 'number of videos', thetas.shape[0])
        database['theta'].append(thetas)
        database['vid_name'].append(vid_names)

    database['theta'] = np.concatenate(database['theta'], axis=0)
    database['vid_name'] = np.concatenate(database['vid_name'], axis=0)

    return database


def read_single_sequence(folder, seq_name, fps=25):
    """
    :param folder: file to store pose sequences
    :param seq_name: Sequence name, i.e. file name
    :param fps: Set the human body attitude speed, the default is 25
    :return: the SMPL structure data in a single sequence
    """
    thetas = []
    vid_names = []

    fname = osp.join(folder + '.npy')

    data = np.load(fname)

    mocap_framerate = int(data['mocap_framerate'])
    sampling_freq = mocap_framerate // fps
    pose = data['poses'][0::sampling_freq, joints_to_use]

    if pose.shape[0] < 60:
        print("false")
        exit()

    shape = np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)
    theta = np.concatenate([pose, shape], axis=1)
    vid_name = np.array([f'{seq_name}'] * pose.shape[0])

    vid_names.append(vid_name)
    thetas.append(theta)

    return np.concatenate(thetas, axis=0), np.concatenate(vid_names, axis=0)


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
        for k in ['root_orient', 'pose_body']:  # , 'betas', 'trans', 'joints']:
            outfname = os.path.join(dataset_dir, split_name, '%s.pt' % k)
            done.append(os.path.exists(outfname))
    return np.all(done)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/HumanML3D')
    args = parser.parse_args()

    db = read_data(args.dir, file_dir='all.txt')
    db_file = osp.join(VIBE_DB_DIR, 'human_ml3d_db.pt')
    print(f'Saving HumanML3D dataset to {db_file}')
    joblib.dump(db, db_file)
