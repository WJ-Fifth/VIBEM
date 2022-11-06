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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_model(generator, gen_optimizer,
               motion_discriminator,
               dis_motion_optimizer,
               performance,
               epoch,
               logdir,
               performance_type,
               best_performance):
    save_dict = {
        'epoch': epoch,
        'gen_state_dict': generator.state_dict(),
        'performance': performance,
        'gen_optimizer': gen_optimizer.state_dict(),
        'disc_motion_state_dict': motion_discriminator.state_dict(),
        'disc_motion_optimizer': dis_motion_optimizer.state_dict(),
    }

    filename = osp.join(logdir, 'checkpoint.pth.tar')
    torch.save(save_dict, filename)

    if performance_type == 'min':
        is_best = performance < best_performance
    else:
        is_best = performance > best_performance

    if is_best:
        logger.info('Best performance achived, saving it!')
        best_performance = performance
        shutil.copyfile(filename, osp.join(logdir, 'model_best.pth.tar'))

        with open(osp.join(logdir, 'best.txt'), 'w') as f:
            f.write(str(float(performance)))


def resume_pretrained(model_path,
                      generator,
                      gen_optimizer,
                      motion_discriminator,
                      dis_motion_optimizer,
                      logger):
    if osp.isfile(model_path):
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['gen_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        best_performance = checkpoint['performance']

        if 'disc_motion_optimizer' in checkpoint.keys():
            motion_discriminator.load_state_dict(checkpoint['disc_motion_state_dict'])
            dis_motion_optimizer.load_state_dict(checkpoint['disc_motion_optimizer'])

        logger.info(f"=> loaded checkpoint '{model_path}' "
                    f"(epoch {start_epoch}, performance {best_performance})")
    else:
        logger.info(f"=> no checkpoint found at '{model_path}'")


def validate(model,
             accumulators,
             test_loader):
    model.eval()

    start = time.time()

    summary_string = ''

    bar = Bar('Validation', fill='#', max=len(test_loader))

    if accumulators is not None:
        for k, v in accumulators.items():
            accumulators[k] = []

    J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

    for i, target in enumerate(test_loader):
        # video = video.to(self.device)
        move_dict_to_device(target, device)

        # <=============
        with torch.no_grad():
            inp = target['features']

            preds = model(inp, J_regressor=J_regressor)

            # convert to 14 keypoint format for evaluation
            # if self.use_spin:
            n_kp = preds[-1]['kp_3d'].shape[-2]
            pred_j3d = preds[-1]['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
            target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
            pred_verts = preds[-1]['verts'].view(-1, 6890, 3).cpu().numpy()
            target_theta = target['theta'].view(-1, 85).cpu().numpy()

            accumulators['pred_verts'].append(pred_verts)
            accumulators['target_theta'].append(target_theta)

            accumulators['pred_j3d'].append(pred_j3d)
            accumulators['target_j3d'].append(target_j3d)
        # =============>

        batch_time = time.time() - start

        summary_string = f'({i + 1}/{len(test_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                         f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

        bar.suffix = summary_string
        bar.next()

    bar.finish()

    logger.info(summary_string)


def evaluate(data_name,
             accumulators):
    for k, v in accumulators.items():
        accumulators[k] = np.vstack(v)

    pred_j3ds = accumulators['pred_j3d']
    target_j3ds = accumulators['target_j3d']

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
    pred_verts = accumulators['pred_verts']
    target_theta = accumulators['target_theta']

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

    if data_name == 'ThreeDPW':
        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'pve': pve,
            'accel': accel,
            'accel_err': accel_err
        }
    elif data_name == 'ssp3d':
        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'pve': pve,
            'pck': pck,
        }
    elif data_name == 'MPII3D':
        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'pck': pck,
        }
    else:
        eval_dict = None
        print(">>>>>>>>>>>>")
        exit()
    log_str = ' '.join([f'{k.upper()}: {v:.4f},' for k, v in eval_dict.items()])
    print(log_str)

    return pa_mpjpe


class Train():
    def __init__(
            self,
            data_loaders,
            val_data_name,
            generator,
            motion_discriminator,
            gen_optimizer,
            dis_motion_optimizer,
            dis_motion_update_steps,
            end_epoch,
            criterion,
            start_epoch=0,
            lr_scheduler=None,
            motion_lr_scheduler=None,
            device=None,
            writer=None,
            debug=False,
            logdir='output',
            resume=None,
            performance_type='min',
            num_iters_per_epoch=1000,
    ):
        self.val_data_name = val_data_name

        # Prepare dataloaders
        self.train_2d_loader, self.train_3d_loader, self.disc_motion_loader, self.valid_loader = data_loaders

        self.disc_motion_iter = iter(self.disc_motion_loader)

        self.train_2d_iter = self.train_3d_iter = None

        if self.train_2d_loader:
            self.train_2d_iter = iter(self.train_2d_loader)

        if self.train_3d_loader:
            self.train_3d_iter = iter(self.train_3d_loader)

        # Models and optimizers
        self.generator = generator
        self.gen_optimizer = gen_optimizer

        self.motion_discriminator = motion_discriminator
        self.dis_motion_optimizer = dis_motion_optimizer

        # Training parameters
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.motion_lr_scheduler = motion_lr_scheduler
        self.device = device
        self.writer = writer
        self.debug = debug
        self.logdir = logdir

        self.dis_motion_update_steps = dis_motion_update_steps

        self.performance_type = performance_type
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')

        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

        self.num_iters_per_epoch = num_iters_per_epoch

        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.logdir)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Resume from a pretrained model
        if resume is not None:
            resume_pretrained(model_path=resume,
                              generator=self.generator,
                              gen_optimizer=self.gen_optimizer,
                              motion_discriminator=self.motion_discriminator,
                              dis_motion_optimizer=self.dis_motion_optimizer,
                              logger=logger)

    def train(self):
        # Single epoch training routine

        losses = AverageMeter()

        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
        }

        self.generator.train()
        self.motion_discriminator.train()

        start = time.time()

        summary_string = ''

        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}', fill='#', max=self.num_iters_per_epoch)

        for i in range(self.num_iters_per_epoch):
            # Dirty solution to reset an iterator
            target_2d = target_3d = None
            if self.train_2d_iter:
                try:
                    target_2d = next(self.train_2d_iter)
                except StopIteration:
                    self.train_2d_iter = iter(self.train_2d_loader)
                    target_2d = next(self.train_2d_iter)

                move_dict_to_device(target_2d, self.device)

            if self.train_3d_iter:
                try:
                    target_3d = next(self.train_3d_iter)
                except StopIteration:
                    self.train_3d_iter = iter(self.train_3d_loader)
                    target_3d = next(self.train_3d_iter)

                move_dict_to_device(target_3d, self.device)

            real_body_samples = real_motion_samples = None

            try:
                real_motion_samples = next(self.disc_motion_iter)
            except StopIteration:
                self.disc_motion_iter = iter(self.disc_motion_loader)
                real_motion_samples = next(self.disc_motion_iter)

            move_dict_to_device(real_motion_samples, self.device)

            # <======= Feedforward generator and discriminator
            if target_2d and target_3d:
                inp = torch.cat((target_2d['features'], target_3d['features']), dim=0).to(self.device)
            elif target_3d:
                inp = target_3d['features'].to(self.device)
            else:
                inp = target_2d['features'].to(self.device)

            timer['data'] = time.time() - start
            start = time.time()

            preds = self.generator(inp)

            timer['forward'] = time.time() - start
            start = time.time()

            gen_loss, motion_dis_loss, loss_dict = self.criterion(
                generator_outputs=preds,
                data_2d=target_2d,
                data_3d=target_3d,
                data_body_mosh=real_body_samples,
                data_motion_mosh=real_motion_samples,
                motion_discriminator=self.motion_discriminator,
            )
            # =======>

            timer['loss'] = time.time() - start
            start = time.time()

            # <======= Backprop generator and discriminator
            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()

            if self.train_global_step % self.dis_motion_update_steps == 0:
                self.dis_motion_optimizer.zero_grad()
                motion_dis_loss.backward()
                self.dis_motion_optimizer.step()
            # =======>

            # <======= Log training info
            total_loss = gen_loss + motion_dis_loss

            losses.update(total_loss.item(), inp.size(0))

            timer['backward'] = time.time() - start
            timer['batch'] = timer['data'] + timer['forward'] + timer['loss'] + timer['backward']
            start = time.time()

            summary_string = f'({i + 1}/{self.num_iters_per_epoch}) | Total: {bar.elapsed_td} | ' \
                             f'ETA: {bar.eta_td:} | loss: {losses.avg:.4f}'

            for k, v in loss_dict.items():
                summary_string += f' | {k}: {v:.2f}'
                self.writer.add_scalar('train_loss/' + k, v, global_step=self.train_global_step)

            for k, v in timer.items():
                summary_string += f' | {k}: {v:.2f}'

            self.writer.add_scalar('train_loss/loss', total_loss.item(), global_step=self.train_global_step)

            if self.debug:
                print('==== Visualize ====')
                from lib.utils.vis import batch_visualize_vid_preds
                video = target_3d['video']
                dataset = 'spin'
                vid_tensor = batch_visualize_vid_preds(video, preds[-1], target_3d.copy(),
                                                       vis_hmr=False, dataset=dataset)
                self.writer.add_video('train-video', vid_tensor, global_step=self.train_global_step, fps=10)

            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next()

            if torch.isnan(total_loss):
                exit('Nan value in loss, exiting!...')
            # =======>

        bar.finish()

        logger.info(summary_string)

    def fit(self):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.train()
            validate(model=self.generator,
                     accumulators=self.evaluation_accumulators,
                     test_loader=self.valid_loader)

            performance = evaluate(data_name=self.val_data_name,
                                   accumulators=self.evaluation_accumulators)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(performance)

            if self.motion_lr_scheduler is not None:
                self.motion_lr_scheduler.step(performance)

            # log the learning rate
            for param_group in self.gen_optimizer.param_groups:
                print(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr/gen_lr', param_group['lr'], global_step=self.epoch)

            for param_group in self.dis_motion_optimizer.param_groups:
                print(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr/dis_lr', param_group['lr'], global_step=self.epoch)

            logger.info(f'Epoch {epoch + 1} performance: {performance:.4f}')

            save_model(generator=self.generator,
                       motion_discriminator=self.motion_discriminator,
                       dis_motion_optimizer=self.dis_motion_optimizer,
                       performance=performance,
                       epoch=epoch,
                       logdir=self.logdir,
                       performance_type=self.performance_type,
                       best_performance=self.best_performance,
                       gen_optimizer=self.gen_optimizer)

            if performance > 100.0:
                exit(f'MPJPE error is {performance}, higher than 100.0. Exiting!...')

        self.writer.close()


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

        self.accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self):
        validate(model=self.model, accumulators=self.accumulators, test_loader=self.test_loader)
        evaluate(data_name=self.data_name, accumulators=self.accumulators)
