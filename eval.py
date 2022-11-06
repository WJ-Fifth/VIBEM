import os
import torch

from lib.dataset import ThreeDPW, ssp3d, MPII3D
from lib.models import VIBE, VIBE_LSTM
from lib.core.main import Evaluator
from lib.core.config import parse_args
from torch.utils.data import DataLoader


def main(cfg):
    print('...Evaluating on {} test set...'.format(cfg.TRAIN.DATASET_EVAL))

    if cfg.MODEL.TEMPORAL_TYPE == 'gru':
        model = VIBE(
            n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            seqlen=cfg.DATASET.SEQLEN,
            hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
            pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
            add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
            bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
            use_residual=cfg.MODEL.TGRU.RESIDUAL,
        ).to(cfg.DEVICE)

    elif cfg.MODEL.TEMPORAL_TYPE == 'lstm':
        model = VIBE_LSTM(
            num_layers=cfg.MODEL.TGRU.NUM_LAYERS,
            batch=cfg.TRAIN.BATCH_SIZE,
            sequences=cfg.DATASET.SEQLEN,
            hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
            pre=cfg.TRAIN.PRETRAINED_REGRESSOR,
            linear=cfg.MODEL.TGRU.ADD_LINEAR,
            bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
            residual=cfg.MODEL.TGRU.RESIDUAL,
        ).to(cfg.DEVICE)

    if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
        best_performance = checkpoint['performance']
        model.load_state_dict(checkpoint['gen_state_dict'])
        print(f'==> Loaded pretrained model from {cfg.TRAIN.PRETRAINED}...')
        if cfg.TRAIN.DATASET_EVAL == 'ThreeDPW':
            print(f'Performance on 3DPW test set {best_performance}')
        elif cfg.TRAIN.DATASET_EVAL == 'ssp3d':
            print(f'Performance on ssp_3D test set {best_performance}')
    else:
        print(f'{cfg.TRAIN.PRETRAINED} is not a pretrained model!!!!')
        exit()

    if cfg.TRAIN.DATASET_EVAL == 'ThreeDPW':
        test_db = ThreeDPW(set='test', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)
    elif cfg.TRAIN.DATASET_EVAL == 'ssp3d':
        test_db = ssp3d(set='test', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)
    elif cfg.TRAIN.DATASET_EVAL == 'MPII3D':
        test_db = MPII3D(set='val', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)
    else:
        print(f'{cfg.DATASET_EVAL} is not a available dataset!!!!')
        exit()

    test_loader = DataLoader(
        dataset=test_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    Evaluator(
        model=model,
        device=cfg.DEVICE,
        test_loader=test_loader,
        data_name=cfg.TRAIN.DATASET_EVAL
    ).run()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()

    main(cfg)
