import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)
    return norm

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

#定义新的混合方式LUmix
class LUMix():
    def __init__(self, mixup_prob=0.1, alpha=0.2, num_classes=10, smoothing=0.1, device='cuda'):
        self.mixup_prob = mixup_prob
        self.alpha = alpha
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.device = device

    def mixup_data(self, x, y):
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mix_x = self.alpha * x + (1 - self.alpha) * x[index, :]
        y_a, y_b = y, y[index]
        return mix_x, y_a, y_b, self.alpha

    def mixup_criterion(self, criterion, pred, y_a, y_b, alpha):
        return alpha * criterion(pred, y_a) + (1 - alpha) * criterion(pred, y_b)

    def smooth_label(self, y_true):
        if self.smoothing == 0:
            return y_true
        confidence = 1 - self.smoothing
        label_shape = torch.Size((y_true.size()[0], self.num_classes))
        y_smoothed = torch.empty(size=label_shape, device=self.device)
        y_smoothed.fill_(self.smoothing / (self.num_classes - 1))
        y_smoothed.scatter_(1, y_true.data.unsqueeze(1), confidence)
        y_smoothed -= (y_smoothed - self.smoothing / (self.num_classes - 1)) * y_true.unsqueeze(1)
        return y_smoothed

    def __call__(self, tgt_imgs, tgt_lbls, imd_imgs, imd_lbls):
        tgt_batch_size = tgt_imgs.size()[0]
        imd_batch_size = imd_imgs.size()[0]
        mixup_tgt_size = int(tgt_batch_size * self.mixup_prob)
        mixup_imd_size = int(imd_batch_size * self.mixup_prob)

        if mixup_tgt_size == 0 and mixup_imd_size == 0:
            return tgt_imgs, tgt_lbls

        if mixup_tgt_size > 0:
            mix_tgt_index = torch.randperm(tgt_batch_size)[:mixup_tgt_size]
            tgt_mix_img = tgt_imgs[mix_tgt_index].to(self.device)
            tgt_mix_lbl = tgt_lbls[mix_tgt_index].to(self.device)
            tgt_mix_img, tgt_mix_lbl_a, tgt_mix_lbl_b, tgt_mix_alpha = self.mixup_data(tgt_mix_img, tgt_mix_lbl)
            tgt_mix_lbl_a = self.smooth_label(tgt_mix_lbl_a)
            tgt_mix_lbl_b = self.smooth_label(tgt_mix_lbl_b)
            tgt_mix_lbl = (tgt_mix_lbl_a, tgt_mix_lbl_b, tgt_mix_alpha)

        if mixup_imd_size > 0:
            mix_imd_index = torch.randperm(imd_batch_size)[:mixup_imd_size]
            imd_mix_img = imd_imgs[mix_imd_index].to(self.device)
            imd_mix_lbl = imd_lbls[mix_imd_index].to(self.device)
            imd_mix_img, imd_mix_lbl_a, imd_mix_lbl_b, imd_mix_alpha = self.mixup_data(imd_mix_img, imd_mix_lbl)
            imd_mix_lbl_a = self.smooth_label(imd_mix_lbl_a)
            imd_mix_lbl_b = self.smooth_label(imd_mix_lbl_b)
            imd_mix_lbl = (imd_mix_lbl_a, imd_mix_lbl_b, imd_mix_alpha)

        if mixup_tgt_size > 0 and mixup_imd_size > 0:
            mixup_tgt_imgs = tgt_imgs[mix_tgt_index].to(self.device)
            mixup_tgt_lbls = [tgt_lbls[mix_tgt_index], tgt_mix_lbl].to(self.device)
            mixup_imd_imgs = imd_imgs[mix_imd_index].to(self.device)
            mixup_imd_lbls = [imd_lbls[mix_imd_index], imd_mix_lbl].to(self.device)
            return torch.cat((mixup_tgt_imgs, mixup_imd_imgs), dim=0), mixup_tgt_lbls, mixup_imd_lbls

        if mixup_tgt_size > 0:
            mixup_tgt_imgs = tgt_imgs[mix_tgt_index].to(self.device)
            mixup_tgt_lbls = [tgt_lbls[mix_tgt_index], tgt_mix_lbl].to(self.device)
            return torch.cat((tgt_imgs, imd_imgs), dim=0), mixup_tgt_lbls, imd_lbls

        if mixup_imd_size > 0:
            mixup_imd_imgs = imd_imgs[mix_imd_index].to(self.device)
            mixup_imd_lbls = [imd_lbls[mix_imd_index], imd_mix_lbl].to(self.device)
            return torch.cat((tgt_imgs, imd_imgs), dim=0), tgt_lbls, mixup_imd_lbls

@UDA.register_module(name='MultiTeacherIMDTGT', force=True)
class MultiTeacherIMDTGT(UDADecorator):
    def __init__(self, **cfg):
        super(MultiTeacherIMDTGT, self).__init__(**cfg)

        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        self.teacher_model_target = build_segmentor(deepcopy(cfg['model']))
        self.teacher_model_imd = build_segmentor(deepcopy(cfg['model']))
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imd_model(self):
        return get_module(self.teacher_model_imd)

    def get_target_model(self):
        return get_module(self.teacher_model_target)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights_imd(self):
        for param in self.get_imd_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_imd_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _init_ema_weights_target(self):
        for param in self.get_target_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_target_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _init_weights(self, module, module_ema):
        for param in module_ema.parameters():
            param.detach_()
        mp = list(module.parameters())
        mcp = list(module_ema.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter, module, module_ema):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(module_ema.parameters(),
                                    module.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log



    def forward_train(self, img, img_metas, gt_semantic_seg=None,
                      imd_img=None, imd_img_metas=None, target_img=None,
                      target_img_metas=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        # Init/update ema model
        if self.local_iter == 0:
            self._init_weights(module=self.get_model(),
                               module_ema=get_module(self.teacher_model_imd))
            self._init_weights(module=self.get_model(),
                               module_ema=get_module(self.teacher_model_target))
            # self._init_ema_weights_imd()
            # self._init_ema_weights_target()
            # assert _params_equal(self.get_ema_model(), self.get_model())
            freeze_model(get_module(self.teacher_model_imd))
            freeze_model(get_module(self.teacher_model_target))
        # 1.Train on source images
        if self.local_iter > 0:
            clean_losses = self.get_model().forward_train(
                img, img_metas, gt_semantic_seg, return_feat=True)
            src_feat = clean_losses.pop('features')
            clean_losses = add_prefix(clean_losses, 'src')
            clean_loss, clean_log_vars = self._parse_losses(clean_losses)
            log_vars.update(clean_log_vars)
            clean_loss.backward(retain_graph=self.enable_fdist)
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                seg_grads = [
                    p.grad.detach().clone() for p in params if p.grad is not None
                ]
                grad_mag = calc_grad_magnitude(seg_grads)
                mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

            # ImageNet feature distance
            if self.enable_fdist:
                feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                          src_feat)
                feat_loss.backward()
                log_vars.update(add_prefix(feat_log, 'src'))
                if self.print_grad_magnitude:
                    params = self.get_model().backbone.parameters()
                    fd_grads = [
                        p.grad.detach() for p in params if p.grad is not None
                    ]
                    fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                    grad_mag = calc_grad_magnitude(fd_grads)
                    mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
            if self.local_iter % 2 == 0:
                # 2.Generate pseudo-label by teacher_imd
                self._update_ema(self.local_iter,
                                 module=self.get_model(),
                                 module_ema=get_module(self.teacher_model_imd))
                for m in get_module(self.teacher_model_imd).modules():
                    if isinstance(m, _DropoutNd):
                        m.training = False
                    if isinstance(m, DropPath):
                        m.training = False
                imd_ema_logits = get_module(self.teacher_model_imd).encode_decode(
                    imd_img, imd_img_metas)
                imd_ema_softmax = torch.softmax(imd_ema_logits.detach(), dim=1)
                imd_pseudo_prob, imd_pseudo_label = torch.max(imd_ema_softmax, dim=1)
                imd_ps_large_p = imd_pseudo_prob.ge(self.pseudo_threshold).long() == 1
                imd_ps_size = np.size(np.array(imd_pseudo_label.cpu()))
                imd_pseudo_weight = torch.sum(imd_ps_large_p).item() / imd_ps_size
                imd_pseudo_weight = imd_pseudo_weight * torch.ones(
                    imd_pseudo_prob.shape, device=dev)

                if self.psweight_ignore_top > 0:
                    # Don't trust pseudo-labels in regions with potential
                    # rectification artifacts. This can lead to a pseudo-label
                    # drift from sky towards building or traffic light.
                    imd_pseudo_weight[:, :self.psweight_ignore_top, :] = 0
                if self.psweight_ignore_bottom > 0:
                    imd_pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
                gt_pixel_weight = torch.ones((imd_pseudo_weight.shape), device=dev)

                # Apply mixing by teacher_imd
                lumix_obj = LUMix(mixup_prob=0.7, alpha=0.2, num_classes=10, smoothing=0.1, device="cuda")

                # 获取混合掩码
                src_imd_mix_masks = get_class_masks(gt_semantic_seg)

                # 源域和中间域的图像、标签以及像素权重
                src_imgs, src_lbls, src_weights = img, gt_semantic_seg[:, 0], gt_pixel_weight
                imd_imgs, imd_lbls, imd_weights = imd_img, imd_pseudo_label, imd_pseudo_weight

                # 将数据划分成多个 batch 进行混合
                batch_size = src_imgs.size(0)
                mixup_batch_size = int(batch_size * lumix_obj.mixup_prob)
                for i in range(0, batch_size, mixup_batch_size):
                    # 获取当前 batch 的数据
                    cur_src_imgs = src_imgs[i:i + mixup_batch_size].clone().detach()
                    cur_src_lbls = src_lbls[i:i + mixup_batch_size].clone().detach()
                    cur_imd_imgs = imd_imgs[i:i + mixup_batch_size].clone().detach()
                    cur_imd_lbls = imd_lbls[i:i + mixup_batch_size].clone().detach()
                    cur_src_weights = src_weights[i:i + mixup_batch_size].clone().detach()
                    cur_imd_weights = imd_weights[i:i + mixup_batch_size].clone().detach()
                    cur_mix_masks = src_imd_mix_masks[i:i + mixup_batch_size].clone().detach()

                    # 将当前 batch 的数据中的源域和中间域进行混合
                    mixed_data, mixed_tgt_lbls, mixed_imd_lbls = lumix_obj(cur_src_imgs, cur_src_lbls, cur_imd_imgs,
                                                                           cur_imd_lbls)
                    mixed_tgt_lbls = mixed_tgt_lbls[0]  # 将混合标签传递给源域的标签

                    # 将混合后的图像和标签分别拷贝到 cur_src_imgs 和 cur_src_lbls 中
                    cur_src_imgs[:mixup_batch_size] = mixed_data[:mixup_batch_size]
                    cur_src_lbls[:mixup_batch_size] = mixed_tgt_lbls.to("cpu")
                    cur_imd_lbls[:mixup_batch_size] = mixed_imd_lbls.to("cpu")

                    # 对源域和中间域的像素权重进行混合
                    mixed_weights, *_ = lumix_obj(cur_src_imgs[:mixup_batch_size], cur_src_lbls[:mixup_batch_size],
                                                  cur_imd_imgs[:mixup_batch_size], cur_imd_lbls[:mixup_batch_size])
                    cur_src_weights[:mixup_batch_size] = mixed_weights.to("cpu")

                    # 对当前 batch 的数据进行强化操作
                    strong_parameters["mix"] = cur_mix_masks.to("cpu")
                    cur_src_imgs, cur_src_lbls, cur_src_weights = strong_transform(strong_parameters, data=cur_src_imgs,
                                                                                   target=cur_src_lbls,
                                                                                   weight=cur_src_weights)
                    cur_imd_imgs, cur_imd_lbls, cur_imd_weights = strong_transform(strong_parameters, data=cur_imd_imgs,
                                                                                   target=cur_imd_lbls,
                                                                                   weight=cur_imd_weights)

                    # 将操作后的数据放回源域和中间域的数据中
                    src_imgs[i:i + mixup_batch_size] = cur_src_imgs.to("cpu")
                    src_lbls[i:i + mixup_batch_size] = cur_src_lbls.to("cpu")
                    src_weights[i:i + mixup_batch_size] = cur_src_weights.to("cpu")
                    imd_imgs[i:i + mixup_batch_size] = cur_imd_imgs.to("cpu")
                    imd_lbls[i:i + mixup_batch_size] = cur_imd_lbls.to("cpu")
                    imd_weights[i:i + mixup_batch_size] = cur_imd_weights.to("cpu")

                # Train on mixed images by teacher_imd
                src_imd_mix_losses = self.get_model().forward_train(
                    src_imgs, img_metas, src_lbls, imd_pseudo_weight, return_feat=True)  # 源域和中间域loss
                src_imd_mix_losses.pop('features')
                src_imd_mix_losses = add_prefix(src_imd_mix_losses, 'src_imd_mix')
                src_imd_mix_loss, src_imd_mix_log_vars = self._parse_losses(src_imd_mix_losses)
                log_vars.update(src_imd_mix_log_vars)
                src_imd_mix_loss.backward()

            if self.local_iter % 2 == 1:  # Learn from teacher_targer as iteration is 2k + 1
                # 3.Generate pseudo-label by teacher_target
                self._update_ema(self.local_iter,
                                 module=self.get_model(),
                                 module_ema=get_module(self.teacher_model_target))
                for m in get_module(self.teacher_model_target).modules():
                    if isinstance(m, _DropoutNd):
                        m.training = False
                    if isinstance(m, DropPath):
                        m.training = False
                ema_logits = get_module(self.teacher_model_target).encode_decode(
                    target_img, target_img_metas)
                ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
                pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
                ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
                ps_size = np.size(np.array(pseudo_label.cpu()))
                pseudo_weight = torch.sum(ps_large_p).item() / ps_size
                pseudo_weight = pseudo_weight * torch.ones(
                    pseudo_prob.shape, device=dev)

                if self.psweight_ignore_top > 0:
                    # Don't trust pseudo-labels in regions with potential
                    # rectification artifacts. This can lead to a pseudo-label
                    # drift from sky towards building or traffic light.
                    pseudo_weight[:, :self.psweight_ignore_top, :] = 0
                if self.psweight_ignore_bottom > 0:
                    pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
                gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

                #target和source混合
                lumix_obj_target = LUMix(mixup_prob=0.7, alpha=0.2, num_classes=10, smoothing=0.1, device="cuda")
                # 获取混合掩码
                mix_masks = get_class_masks(gt_semantic_seg)

                # 源域和中间域的图像、标签以及像素权重
                src_img, src_lbl, src_weights = img, gt_semantic_seg[:, 0], gt_pixel_weight
                target_imgs, target_lbls, target_weights = target_img, pseudo_label, pseudo_weight

                # 将数据划分成多个 batch 进行混合
                batch_size = src_img.size(0)
                mixup_batch_size = int(batch_size * lumix_obj_target.mixup_prob)
                for i in range(0, batch_size, mixup_batch_size):
                    # 获取当前 batch 的数据
                    cur_src_imgs = src_img[i:i + mixup_batch_size].clone().detach()
                    cur_src_lbls = src_lbl[i:i + mixup_batch_size].clone().detach()
                    cur_target_imgs = target_imgs[i:i + mixup_batch_size].clone().detach()
                    cur_target_lbls = target_lbls[i:i + mixup_batch_size].clone().detach()
                    cur_src_weights = src_weights[i:i + mixup_batch_size].clone().detach()
                    cur_imd_weights = target_weights[i:i + mixup_batch_size].clone().detach()
                    cur_mix_masks = mix_masks[i:i + mixup_batch_size].clone().detach()

                    # 将当前 batch 的数据中的源域和中间域进行混合
                    mixed_data, mixed_tgt_lbls, mixed_target_lbls = lumix_obj_target(cur_src_imgs, cur_src_lbls, cur_target_imgs,
                                                                           cur_target_lbls)
                    mixed_tgt_lbls = mixed_tgt_lbls[0]  # 将混合标签传递给源域的标签

                    # 将混合后的图像和标签分别拷贝到 cur_src_imgs 和 cur_src_lbls 中
                    cur_src_imgs[:mixup_batch_size] = mixed_data[:mixup_batch_size]
                    cur_src_lbls[:mixup_batch_size] = mixed_tgt_lbls.to("cpu")
                    cur_target_lbls[:mixup_batch_size] = mixed_target_lbls.to("cpu")

                    # 对源域和中间域的像素权重进行混合
                    mixed_weights, *_ = lumix_obj_target(cur_src_imgs[:mixup_batch_size], cur_src_lbls[:mixup_batch_size],
                                                  cur_target_imgs[:mixup_batch_size], cur_target_lbls[:mixup_batch_size])
                    cur_src_weights[:mixup_batch_size] = mixed_weights.to("cpu")

                    # 对当前 batch 的数据进行强化操作
                    strong_parameters["mix"] = cur_mix_masks.to("cpu")
                    cur_src_imgs, cur_src_lbls, cur_src_weights = strong_transform(strong_parameters, data=cur_src_imgs,
                                                                                   target=cur_src_lbls,
                                                                                   weight=cur_src_weights)
                    cur_imd_imgs, cur_imd_lbls, cur_target_weights = strong_transform(strong_parameters, data=cur_target_imgs,
                                                                                   target=cur_target_lbls,
                                                                                   weight=cur_target_weights)

                    # 将操作后的数据放回源域和中间域的数据中
                    src_img[i:i + mixup_batch_size] = cur_src_imgs.to("cpu")
                    src_lbl[i:i + mixup_batch_size] = cur_src_lbls.to("cpu")
                    src_weights[i:i + mixup_batch_size] = cur_src_weights.to("cpu")
                    target_imgs[i:i + mixup_batch_size] = cur_target_imgs.to("cpu")
                    target_lbls[i:i + mixup_batch_size] = cur_target_lbls.to("cpu")
                    target_weights[i:i + mixup_batch_size] = cur_target_weights.to("cpu")


                # Train on mixed images
                mix_losses = self.get_model().forward_train(
                    src_img, img_metas, src_lbl, pseudo_weight, return_feat=True)
                mix_losses.pop('features')
                mix_losses = add_prefix(mix_losses, 'mix')
                mix_loss, mix_log_vars = self._parse_losses(mix_losses)
                log_vars.update(mix_log_vars)
                mix_loss.backward()

                if self.local_iter % self.debug_img_interval == 0:
                    out_dir = os.path.join(self.train_cfg['work_dir'],
                                           'class_mix_debug')
                    os.makedirs(out_dir, exist_ok=True)
                    vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
                    vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
                    vis_mixed_img = torch.clamp(denorm(src_img, means, stds), 0, 1)
                    for j in range(batch_size):
                        rows, cols = 2, 5
                        fig, axs = plt.subplots(
                            rows,
                            cols,
                            figsize=(3 * cols, 3 * rows),
                            gridspec_kw={
                                'hspace': 0.1,
                                'wspace': 0,
                                'top': 0.95,
                                'bottom': 0,
                                'right': 1,
                                'left': 0
                            },
                        )
                        subplotimg(axs[0][0], vis_img[j], 'Source Image')
                        subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                        subplotimg(
                                axs[0][1],
                                gt_semantic_seg[j],
                                'Source Seg GT',
                                cmap='cityscapes')
                        subplotimg(
                                axs[1][1],
                                pseudo_label[j],
                                'Target Seg (Pseudo) GT',
                                cmap='cityscapes')
                        subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                        subplotimg(
                                axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                            # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                            #            cmap="cityscapes")
                        subplotimg(
                                axs[1][3], src_lbl[j], 'Seg Target', cmap='cityscapes')
                        subplotimg(
                                axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                        if self.debug_fdist_mask is not None:
                                subplotimg(
                                    axs[0][4],
                                    self.debug_fdist_mask[j][0],
                                    'FDist Mask',
                                    cmap='gray')
                        if self.debug_gt_rescale is not None:
                                subplotimg(
                                    axs[1][4],
                                    self.debug_gt_rescale[j],
                                    'Scaled GT',
                                    cmap='cityscapes')
                        for ax in axs.flat:
                                ax.axis('off')
                        plt.savefig(
                                os.path.join(out_dir,
                                             f'{(self.local_iter + 1):06d}_{j}.png'))
                        plt.close()
        self.local_iter += 1

        return log_vars