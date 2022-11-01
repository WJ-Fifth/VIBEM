# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar

from lib.core.config import VIBE_DATA_DIR
from lib.utils.utils import move_dict_to_device, AverageMeter

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)


class Evaluator():
    def __init__(
            self,
            test_loader,
            model,
            device=None,
            data_name=None,
    ):
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.data_name = data_name

        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def validate(self):
        self.model.eval()

        start = time.time()

        summary_string = ''

        bar = Bar('Validation', fill='#', max=len(self.test_loader))

        if self.evaluation_accumulators is not None:
            for k, v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

        for i, target in enumerate(self.test_loader):
            # video = video.to(self.device)
            move_dict_to_device(target, self.device)

            # <=============
            with torch.no_grad():
                inp = target['features']

                preds = self.model(inp, J_regressor=J_regressor)

                # convert to 14 keypoint format for evaluation
                # if self.use_spin:
                n_kp = preds[-1]['kp_3d'].shape[-2]
                pred_j3d = preds[-1]['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                pred_verts = preds[-1]['verts'].view(-1, 6890, 3).cpu().numpy()
                target_theta = target['theta'].view(-1, 85).cpu().numpy()

                self.evaluation_accumulators['pred_verts'].append(pred_verts)
                self.evaluation_accumulators['target_theta'].append(target_theta)

                self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
                self.evaluation_accumulators['target_j3d'].append(target_j3d)
            # =============>

            batch_time = time.time() - start

            summary_string = f'({i + 1}/{len(self.test_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                             f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

            bar.suffix = summary_string
            bar.next()

        bar.finish()

        logger.info(summary_string)

    def evaluate(self):

        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
        target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0

        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis

        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        pred_verts = self.evaluation_accumulators['pred_verts']
        target_theta = self.evaluation_accumulators['target_theta']

        m2mm = 1000

        pve = np.mean(compute_error_verts(target_theta=target_theta, pred_verts=pred_verts)) * m2mm
        accel = np.mean(compute_accel(pred_j3ds)) * m2mm
        accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm

        def cal_distance(preds, targets):
            dis = np.abs(preds - targets)
            return dis

        def compute_similarity_transform(source_points, target_points):
            """Computes a similarity transform (sR, t) that takes a set of 3D points
            source_points (N x 3) closest to a set of 3D points target_points, where R
            is an 3x3 rotation matrix, t 3x1 translation, s scale. And return the
            transformed 3D points source_points_hat (N x 3). i.e. solves the orthogonal
            Procrutes problem.
            Note:
                Points number: N
            Args:
                source_points (np.ndarray): Source point set with shape [N, 3].
                target_points (np.ndarray): Target point set with shape [N, 3].
            Returns:
                np.ndarray: Transformed source point set with shape [N, 3].
            """

            assert target_points.shape[0] == source_points.shape[0]
            assert target_points.shape[1] == 3 and source_points.shape[1] == 3

            source_points = source_points.T
            target_points = target_points.T

            # 1. Remove mean.
            mu1 = source_points.mean(axis=1, keepdims=True)
            mu2 = target_points.mean(axis=1, keepdims=True)
            X1 = source_points - mu1
            X2 = target_points - mu2

            # 2. Compute variance of X1 used for scale.
            var1 = np.sum(X1 ** 2)

            # 3. The outer product of X1 and X2.
            K = X1.dot(X2.T)

            # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
            # singular vectors of K.
            U, _, Vh = np.linalg.svd(K)
            V = Vh.T
            # Construct Z that fixes the orientation of R to get det(R)=1.
            Z = np.eye(U.shape[0])
            Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
            # Construct R.
            R = V.dot(Z.dot(U.T))

            # 5. Recover scale.
            scale = np.trace(R.dot(K)) / var1

            # 6. Recover translation.
            t = mu2 - scale * (R.dot(mu1))

            # 7. Transform the source points:
            source_points_hat = scale * R.dot(source_points) + t

            source_points_hat = source_points_hat.T

            return source_points_hat

        def keypoint_3d_pck(pred, gt, mask, alignment='none', threshold=0.15):
            assert mask.any()

            if alignment == 'none':
                pass
            elif alignment == 'procrustes':
                pred = np.stack([
                    compute_similarity_transform(pred_i, gt_i)
                    for pred_i, gt_i in zip(pred, gt)
                ])
            elif alignment == 'scale':
                pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
                pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
                scale_factor = pred_dot_gt / pred_dot_pred
                pred = pred * scale_factor[:, None, None]
            else:
                raise ValueError(f'Invalid value for alignment: {alignment}')

            error = np.linalg.norm(pred - gt, ord=2, axis=-1)
            pck = (error < threshold).astype(np.float32)[mask].mean() * 100

            return pck

        mask = np.ones((target_j3ds.shape[0], target_j3ds.shape[1]))
        mask = np.array(mask, dtype=bool)
        pck = keypoint_3d_pck(pred_j3ds, target_j3ds, mask=mask)

        if self.data_name == 'ThreeDPW':
            eval_dict = {
                'mpjpe': mpjpe,
                'pa-mpjpe': pa_mpjpe,
                'pve': pve,
                'accel': accel,
                'accel_err': accel_err
            }
        elif self.data_name == 'ssp3d':
            eval_dict = {
                'mpjpe': mpjpe,
                'pa-mpjpe': pa_mpjpe,
                'pve': pve,
                'pck': pck,
            }
        elif self.data_name == 'MPII3D':
            eval_dict = {
                'mpjpe': mpjpe,
                'pa-mpjpe': pa_mpjpe,
                'pck': pck,
            }
        else:
            print(">>>>>>>>>>>>")
            exit()
        log_str = ' '.join([f'{k.upper()}: {v:.4f},' for k, v in eval_dict.items()])
        print(log_str)

    def run(self):
        self.validate()
        self.evaluate()
