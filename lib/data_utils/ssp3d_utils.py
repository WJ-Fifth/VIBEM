# -*- coding: utf-8 -*-
# By JInwu Wang u7354172
#

import sys

sys.path.append('.')

import os
import cv2
import torch
import joblib
import argparse
import numpy as np
import pickle as pkl
import os.path as osp
from tqdm import tqdm

from lib.models import spin, backbone
from lib.data_utils.kp_utils import *
from lib.core.config import VIBE_DB_DIR, VIBE_DATA_DIR
from lib.utils.smooth_bbox import get_smooth_bbox_params
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from lib.data_utils.feature_extractor import extract_features
from lib.utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis

NUM_JOINTS = 24
VIS_THRESH = 0.3
MIN_KP = 6


def read_data(folder, backbone_name):
    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'img_name': [],
        'features': [],
        # 'valid': [],
    }
    if backbone_name == 'resnet50' or backbone_name == 'resnext50_32x4d' or backbone_name == 'swin':
        model = backbone.Backbone(backbone_name)
    elif backbone_name == 'spin':
        model = spin.get_pretrained_hmr()

    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)

    data_file = osp.join(folder, 'labels.npz')

    data = np.load(data_file)

    img_dir = osp.join(folder, 'images')
    silhouettes_dir = osp.join(folder, 'silhouettes')

    num_people = len(data['poses'])
    num_frames = len(data['fnames'])
    # print(data['fnames'][0])

    assert (data['joints2D'].shape[0] == num_frames)

    pose = torch.from_numpy(data['poses']).float()
    shape = torch.from_numpy(data['shapes']).float()
    trans = torch.from_numpy(data['cam_trans']).float()

    gender = data['genders']

    bbox_centre = data['bbox_centres']
    bbox_whs = data['bbox_whs']

    j2d = data['joints2D']

    output = smpl(betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3], transl=trans)
    j3d = output.joints

    bbox = np.vstack([bbox_centre[:, 0], bbox_centre[:, 1], bbox_whs, bbox_whs]).T

    bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2d, vis_thresh=VIS_THRESH, sigma=8)

    img_paths = []
    vid_names = []
    for i_fname in data['fnames']:
        img_path = osp.join(img_dir, i_fname)
        img_paths.append(img_path)
        name_split = i_fname.split('_')
        if name_split[3] == 'clip':
            vid_names.append(name_split[0] + '_vid_' + name_split[2] + '_clip_' + name_split[4] + '_person_' + name_split[6])
        else:
            vid_names.append(name_split[0]+'_'+name_split[1]+'_vid_' +
                             name_split[3] + '_clip_' + name_split[5] + '_person_' + name_split[7])

    img_paths_array = np.array(img_paths)[time_pt1:time_pt2]
    dataset['vid_name'] = np.array(vid_names)[time_pt1:time_pt2]
    # print(dataset['vid_name'].shape)
    dataset['frame_id'] = data['fnames'][time_pt1:time_pt2]
    dataset['img_name'] = img_paths_array
    dataset['joints3D'] = j3d.numpy()
    dataset['joints2D'] = j2d
    dataset['shape'] = shape.numpy()
    dataset['pose'] = pose.numpy()
    dataset['bbox'] = bbox
    # dataset['valid'].append(campose_valid[time_pt1:time_pt2])

    features = extract_features(model, img_paths_array, bbox, kp_2d=j2d[time_pt1:time_pt2], debug=debug,
                                dataset='ssp3d', scale=1.2, model_name=backbone_name)
    dataset['features'] = features

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/ssp_3d')
    args = parser.parse_args()
    debug = False

    dataset = read_data(folder=args.dir, backbone_name='spin')
    # print(dataset['vid_name'])
    # exit()
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'ssp3d_test_db.pt'))
    print("ssp3d_test_db set success !!!")
