from typing import List
import os
import yaml
import argparse
import time
from scipy import ndimage
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import torch.nn as nn
import random
from torch.utils.data import DataLoader

import models
from dataset import blind_SegDataset
from metric import Measurement
from loss import DiceLoss, PerceptualLoss
import torch
import torch.nn.functional as F
# 시드 설정
seed_value = 107

# PyTorch 시드 설정
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Python 시드 설정
random.seed(seed_value)

# Numpy 시드 설정
np.random.seed(seed_value)

class WeightedCrossEntropyLoss(nn.Module):
    """
    가중치가 적용된 CrossEntropyLoss
    가려진 영역 근처의 object 밀도와 위치를 고려하여 가중치를 계산
    """

    def __init__(self, num_classes=3, base_weight=1.0, density_weight=2.0, position_weight=1.5,
                 density_kernel_size=15, position_decay_factor=0.8):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.base_weight = base_weight
        self.density_weight = density_weight
        self.position_weight = position_weight
        self.density_kernel_size = density_kernel_size
        self.position_decay_factor = position_decay_factor

    def create_object_density_map(self, target, mask):
        """
        Object 밀도 맵 생성 - 가려진 영역 근처에서 object 밀도 계산
        Args:
            target: GT mask (B, H, W)
            mask: binary mask (B, 1, H, W) - 가려진 영역
        Returns:
            density_map: Object 밀도 맵 (B, H, W)
        """
        batch_size, height, width = target.shape

        # Object 존재 여부 (crop + weed)
        object_mask = (target > 0).float()  # (B, H, W)

        # 간단한 평균 필터를 사용하여 밀도 맵 생성
        kernel_size = min(self.density_kernel_size, height, width)
        if kernel_size % 2 == 0:
            kernel_size -= 1  # 홀수로 만들기

        # 평균 필터 kernel 생성
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=target.device)
        kernel = kernel / (kernel_size * kernel_size)

        # Convolution으로 밀도 맵 생성
        object_mask = object_mask.unsqueeze(1)  # (B, 1, H, W)
        padding = kernel_size // 2

        density_map = F.conv2d(object_mask, kernel, padding=padding)
        density_map = density_map.squeeze(1)  # (B, H, W)

        # 크기가 정확히 맞는지 확인하고 필요시 조정
        if density_map.shape != target.shape:
            density_map = F.interpolate(density_map.unsqueeze(1), size=target.shape[1:], mode='bilinear',
                                        align_corners=False).squeeze(1)

        return density_map

    def create_position_weight_map(self, mask, height, width, device):
        """
        위치 기반 가중치 맵 생성
        가려진 영역 위쪽으로 갈수록 가중치 증가 (지수적 감소)
        """
        mask_2d = mask.squeeze(1)  # (B, H, W)
        position_weights = torch.ones(mask_2d.shape, device=device)

        for b in range(mask_2d.shape[0]):
            mask_row = mask_2d[b]  # (H, W)

            # 각 열에서 가려진 영역의 시작 y 좌표 찾기
            for x in range(width):
                # 위에서부터 가려진 영역 시작점 찾기
                masked_indices = torch.where(mask_row[:, x] == 1)[0]
                if len(masked_indices) > 0:
                    mask_start_y = masked_indices[0].item()

                    # 가려진 영역 위쪽에 가중치 적용 (지수적 증가)
                    for y in range(mask_start_y):
                        # 위쪽으로 갈수록 가중치 증가 (지수적 증가)
                        distance_from_mask = mask_start_y - y
                        weight_factor = 1.0 + self.position_weight * (self.position_decay_factor ** distance_from_mask)
                        position_weights[b, y, x] = weight_factor

        return position_weights

    def create_combined_weight_map(self, target, mask):
        """
        Object 밀도와 위치를 결합한 최종 가중치 맵 생성
        """
        batch_size, height, width = target.shape

        # 1. Object 밀도 맵 생성
        density_map = self.create_object_density_map(target, mask)

        # 2. 위치 기반 가중치 맵 생성
        position_weights = self.create_position_weight_map(mask, height, width, target.device)

        # 3. 텐서 크기 검증 및 조정
        if density_map.shape != target.shape:
            density_map = F.interpolate(density_map.unsqueeze(1), size=target.shape[1:], mode='bilinear',
                                        align_corners=False).squeeze(1)

        if position_weights.shape != target.shape:
            position_weights = F.interpolate(position_weights.unsqueeze(1), size=target.shape[1:], mode='bilinear',
                                             align_corners=False).squeeze(1)

        # 4. 최종 가중치 계산
        # 밀도가 높고 위치 가중치가 높은 곳에 더 큰 가중치
        final_weights = self.base_weight + self.position_weight * position_weights + self.density_weight * density_map

        # 가려진 영역 내부는 기본 가중치만 적용
        mask_2d = mask.squeeze(1)
        final_weights = torch.where(mask_2d == 1, self.base_weight, final_weights)

        return final_weights, density_map, position_weights

    def forward(self, pred, target, mask):
        # 가중치 맵 생성
        final_weights, density_map, position_weights = self.create_combined_weight_map(target, mask)

        # CrossEntropyLoss 계산 (가중치 적용)
        loss = F.cross_entropy(pred, target, reduction='none')  # (B, H, W)

        # 가중치 적용
        weighted_loss = loss * final_weights

        return weighted_loss.mean(), final_weights, density_map, position_weights

class OFKD(nn.Module):

    def __init__(self,
                 bg_id: int = 0,
                 radius: int = 10,
                 mode: str = 'gauss',
                 lambda_b: float = 1.0,
                 gamma: float = 0.1,
                 temperature: float = 4.0,
                 device: str = 'cuda'):
        super(OFKD, self).__init__()

        self.bg_id = bg_id
        self.radius = radius
        self.mode = mode
        self.lambda_b = lambda_b
        self.gamma = gamma
        self.temperature = temperature
        self.device = device

    # -------------------------------------------------------------
    def compute_distance_weight(self, teacher_probs: torch.Tensor) -> torch.Tensor:
        tp = teacher_probs.detach().cpu().numpy()  # (B, C, H, W)
        B, C, H, W = tp.shape

        weights = np.zeros((B, H, W), dtype=np.float32)
        sigma = self.radius / 2.0

        for b in range(B):
            pred = tp[b].argmax(axis=0).astype(np.uint8)  # (H, W)
            fg = (pred != self.bg_id).astype(np.uint8)

            if fg.sum() == 0:
                continue

            # object mask 주변 distance
            dist = ndimage.distance_transform_edt(1 - fg)
            mask_inside = (dist <= self.radius).astype(np.float32)

            if self.mode == 'gauss':
                wdist = np.exp(-(dist**2) / (2 * sigma**2))
            else:
                wdist = np.clip(1 - dist / self.radius, 0.0, 1.0)

            wdist *= mask_inside

            eroded = ndimage.binary_erosion(
                fg,
                structure=ndimage.generate_binary_structure(2, 1)
            )
            boundary = (fg - eroded).astype(np.float32)
            wb = wdist * (1.0 + self.lambda_b * boundary)

            w_final = wb

            weights[b] = w_final * mask_inside

        return torch.from_numpy(weights).float().to(self.device)

    # -------------------------------------------------------------
    def weighted_kd(self,
                    student_logits: torch.Tensor,
                    teacher_logits: torch.Tensor,
                    weight_map: torch.Tensor,
                    eps: float = 1e-8) -> torch.Tensor:

        T = self.temperature

        with torch.no_grad():
            t_prob_T = F.softmax(teacher_logits / T, dim=1)   # (B, C, H, W)

            entropy = -(t_prob_T * (t_prob_T.clamp_min(eps).log())).sum(dim=1)  # (B, H, W)
            ent_max = entropy.amax(dim=(1, 2), keepdim=True)
            ent_norm = entropy / (ent_max + eps)              # 0~1
            conf_weight = 1.0 - ent_norm                      # high conf → 1, low conf → 0 근처

            conf_factor = self.gamma + (1.0 - self.gamma) * conf_weight  # (B, H, W)

            extra_weight = conf_factor

        s_logp = F.log_softmax(student_logits / T, dim=1)
        t_prob = F.softmax(teacher_logits.detach() / T, dim=1)

        kl_per_pixel = (t_prob * (torch.log(t_prob + eps) - s_logp)).sum(dim=1)  # (B, H, W)

        total_weight = weight_map * extra_weight  # (B, H, W)

        loss = (total_weight * kl_per_pixel).sum() / (total_weight.sum() + eps)
        loss = (T * T) * loss  # T^2 scaling

        return loss

    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor):

        # 내부 softmax (T=1)로 거리/경계용 weight map 계산
        teacher_probs = F.softmax(teacher_logits, dim=1)
        weight_map = self.compute_distance_weight(teacher_probs)

        return self.weighted_kd(student_logits, teacher_logits, weight_map)

class ObjectAwareChannelKD(nn.Module):
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                student_feat: torch.Tensor,   # (B, C, H', W')
                teacher_feat: torch.Tensor,   # (B, C, H', W')
                weight_map: torch.Tensor,     # (B, H, W) - logits 해상도 기준
                eps: float = 1e-8):

        B, C, Hf, Wf = student_feat.shape
        _, Hm, Wm = weight_map.shape

        # (B, 1, H, W) -> encoder 해상도로 downsample
        w = weight_map.unsqueeze(1).float()          # (B, 1, H, W)
        w_ds = F.interpolate(w, size=(Hf, Wf), mode='bilinear', align_corners=False)
        w_ds = w_ds.clamp(min=0.0)
        w_sum = w_ds.sum(dim=(2, 3), keepdim=True) + eps   # (B, 1, 1, 1)

        # object-aware weighted GAP
        F_t = teacher_feat.detach()  # teacher는 gradient X
        F_s = student_feat

        t_weighted = F_t * w_ds
        s_weighted = F_s * w_ds

        # 채널별 평균 벡터: (B, C)
        v_t = t_weighted.sum(dim=(2, 3)) / w_sum.squeeze(1).squeeze(1)  # (B, C)
        v_s = s_weighted.sum(dim=(2, 3)) / w_sum.squeeze(1).squeeze(1)  # (B, C)

        # Temperature-scaled channel KD
        T = self.temperature
        t_prob = F.softmax(v_t / T, dim=1)             # (B, C)
        s_logprob = F.log_softmax(v_s / T, dim=1)      # (B, C)

        kl = (t_prob * (torch.log(t_prob + eps) - s_logprob)).sum(dim=1)  # (B,)
        loss = kl.mean() * (T * T)
        return loss


class Trainer():
    def __init__(self, opt, cfg, model_T, model):
        print(opt)
        print(cfg)

        self.model_T = model_T
        self.model = model
        self.start_epoch = 0
        self.num_epochs = cfg['NUM_EPOCHS']
        self.device = cfg['GPU']
        self.num_classes = cfg['NUM_CLASSES']
        self.KD_weight = 1
        self.T = 4
        self.neck_trans = nn.Conv2d(256, 512, kernel_size=1) #MLP(256, 512)#

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(cfg['OPTIM']['LR_INIT']), betas=(0.5, 0.999))
        self.optimizer.add_param_group({'params': self.neck_trans.parameters()})

        # data load
        train_dataset = blind_SegDataset(os.path.join(cfg['DATA_DIR'], 'train'), resize=cfg['RESIZE'], targetresize=True, randomaug=True, direction='top', cover_percent=0.1)
        val_dataset = blind_SegDataset(os.path.join(cfg['DATA_DIR'], 'val'), resize=cfg['RESIZE'], targetresize=True, direction='top', cover_percent=0.1)

        self.trainloader = DataLoader(train_dataset, cfg['BATCH_SIZE'], shuffle=True, drop_last=True)
        self.valloader = DataLoader(val_dataset, 1, shuffle=False)

        self.loss = WeightedCrossEntropyLoss(
            num_classes=self.num_classes,
            base_weight=opt.base_weight,
            density_weight=opt.density_weight,
            position_weight=opt.position_weight,
            density_kernel_size=opt.density_kernel_size,
            position_decay_factor=opt.position_decay_factor
        )
        self.loss_kd = OFKD()
        self.loss_kd2 = ObjectAwareChannelKD()
        self.measurement = Measurement(self.num_classes)

        # if resume
        if cfg['LOAD_WEIGHTS'] != '':
            print('############# resume training #############')
            self.resume = True
            self.start_epoch, optimizer_statedict, self.best_miou = self.load_checkpoint(cfg['LOAD_WEIGHTS'])
            self.optimizer.load_state_dict(optimizer_statedict)
            self.save_dir = os.path.split(os.path.split(cfg['LOAD_WEIGHTS'])[0])[0]
            self.ckpoint_path = os.path.join(self.save_dir, 'ckpoints')
            self.best_miou_test = 0
        else:
            # save path
            self.resume = False
            os.makedirs(cfg['SAVE_DIR'], exist_ok=True)
            train_name = os.path.basename(os.path.join(cfg['DATA_DIR'], 'train'))
            fold_name = os.path.basename(cfg['DATA_DIR'])
            self.save_dir = os.path.join(cfg['SAVE_DIR'],
                                         f'{self.model.__class__.__name__}-ep{self.num_epochs}-{train_name}-{fold_name}-' + str(
                                             len(os.listdir(cfg['SAVE_DIR']))))
            self.ckpoint_path = os.path.join(self.save_dir, 'ckpoints')
            os.makedirs(self.ckpoint_path)

    def train(self, opt):
        # 메모리 관리 개선
        torch.cuda.empty_cache()
        gc.collect()

        if not self.resume:
            self.device_setting(self.device)
            self.model_T.load_state_dict(torch.load(opt.weight_T)['network'])
            self.best_miou = 0
            self.best_miou_test = 0
        if opt.save_img:
            os.makedirs(os.path.join(self.save_dir, 'val_imgs'), exist_ok=True)
        if opt.save_txt:
            self.f = open(os.path.join(self.save_dir, 'result.txt'), 'a')
        if opt.save_graph:
            loss_list = []
            self.val_loss_list = []
            train_loss_kd_list = []
            train_loss_seg_list = []
            self.val_loss_kd_list = []
            self.val_loss_seg_list = []

        if opt.save_csv:
            loss_list = []
            self.val_loss_list = []
            miou_list, lr_list = [], []
            self.val_miou_list = []

        self.best_miou_epoch = 0
        print('######### start training #########')
        for epoch in range(self.start_epoch, self.num_epochs):

            self.model = self.model.to(self.device)
            self.model_T = self.model_T.to(self.device)
            self.neck_trans = self.neck_trans.to(self.device)

            ep_start = time.time()
            epoch_loss = 0
            epoch_loss_kd, epoch_loss_seg = 0, 0
            epoch_miou = 0
            iou_per_class = np.array([0] * (self.num_classes), dtype=np.float64)
            self.model.train()
            self.model_T.eval()
            self.neck_trans.train()
            trainloader_len = len(self.trainloader)
            self.start_timer()

            for i, data in enumerate(tqdm(self.trainloader), 0):
                img, input_img, target_img = data[:3]
                b_mask = data[-2]
                label_img = self.mask_labeling(target_img, self.num_classes)

                input_img, label_img = input_img.to(self.device), label_img.to(self.device)
                b_mask = b_mask.to(self.device)

                self.optimizer.zero_grad()

                concat_input = torch.cat((input_img, b_mask), dim=1)

                with torch.no_grad():
                    pred_T , _, _, neck_T = self.model_T(concat_input)

                # predict
                pred, neck = self.model(concat_input)

                # loss
                kd_weight_map = self.loss_kd.compute_distance_weight(pred_T)
                loss_kd = self.loss_kd2(self.neck_trans(neck),neck_T, kd_weight_map) +self.loss_kd(pred,pred_T)# +

                loss_seg, weight_map, density_map, position_weights = self.loss(pred, label_img, b_mask)  # [in_index].mean()
                loss_output = loss_seg  + self.KD_weight*loss_kd
                loss_output.backward()
                self.optimizer.step()

                pred_numpy, label_numpy = pred.detach().cpu().numpy(), label_img.detach().cpu().numpy()
                epoch_loss += loss_output.item()
                epoch_loss_kd += loss_kd.item()
                epoch_loss_seg += loss_seg.item()

                _, ep_miou, ious, _, _, _ = self.measurement(pred_numpy, label_numpy)
                epoch_miou += ep_miou
                iou_per_class += ious

            epoch_loss /= trainloader_len
            epoch_loss_kd /= trainloader_len
            epoch_loss_seg /= trainloader_len
            epoch_miou /= trainloader_len
            epoch_ious = np.round((iou_per_class / trainloader_len), 5).tolist()

            if opt.save_graph:
                loss_list += [epoch_loss]
                train_loss_kd_list += [epoch_loss_kd]
                train_loss_seg_list += [epoch_loss_seg]

            if opt.save_csv:
                if not opt.save_graph: loss_list += [epoch_loss]
                miou_list += [epoch_miou]
                lr_list += [self.optimizer.param_groups[0]['lr']]

            traintxt = f"[epoch {epoch} Loss: {epoch_loss:.4f}, LearningRate :{self.optimizer.param_groups[0]['lr']:.6f}, trainmIOU: {epoch_miou}, train IOU per class:{epoch_ious}, time: {(time.time() - ep_start):.4f} sec \n"
            print(traintxt)

            if opt.save_txt:
                self.f.write(traintxt)

            # save model
            self.save_checkpoint('model_last.pth', self.model, epoch)

            # validation
            self.val_test(epoch, opt)

            # 주기적인 메모리 정리
            if epoch % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        if opt.save_graph:
            self.save_lossgraph(loss_list, self.val_loss_list)
            self.save_lossgraph2(train_loss_kd_list, train_loss_seg_list)
            self.save_lossgraph3(self.val_loss_kd_list,  self.val_loss_seg_list)
        if opt.save_csv:
            self.save_csv('train', [loss_list, lr_list, miou_list], 'training.csv')
            self.save_csv('val', [self.val_loss_list, self.val_miou_list], 'validation.csv')

        print("----- train finish -----")
        self.end_timer_and_print()

    def device_setting(self, device):
        if device != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda:' + device)
        else:
            self.device = torch.device('cpu')

    def val_test(self, epoch, opt):
        self.model_T.eval()
        self.model.eval()
        self.neck_trans.eval()

        val_miou, val_loss = 0, 0
        val_loss_kd, val_loss_seg = 0, 0
        iou_per_class = np.array([0] * (self.num_classes), dtype=np.float64)
        val_loss_list = []

        for i, data in enumerate(tqdm(self.valloader), 0):
            filename = data[-1]
            img, input_img, target_img = data[:3]
            b_mask = data[-2]
            label_img = self.mask_labeling(target_img, self.num_classes)
            input_img, label_img = input_img.to(self.device), label_img.to(self.device)
            b_mask = b_mask.to(self.device)

            with torch.no_grad():
                concat_input = torch.cat((input_img, b_mask), dim=1)
                pred_T, _, _, neck_T = self.model_T(concat_input)
                pred, neck = self.model(concat_input)

                kd_weight_map = self.loss_kd.compute_distance_weight(pred_T)
                loss_kd = self.loss_kd2(self.neck_trans(neck), neck_T, kd_weight_map) + self.loss_kd(pred, pred_T)# +
                loss_seg, weight_map, density_map, position_weights = self.loss(pred, label_img, b_mask)
                loss_output = loss_seg  + self.KD_weight*loss_kd

                val_loss += loss_output.item()
                val_loss_kd += loss_kd.item()
                val_loss_seg += loss_seg.item()

            pred_numpy, label_numpy = pred.detach().cpu().numpy(), label_img.detach().cpu().numpy()
            # ob_numpy = ob.detach().cpu().numpy()
            _, ep_miou, ious, _, _, _ = self.measurement(pred_numpy, label_numpy)
            val_miou += ep_miou
            iou_per_class += ious

            if opt.save_img:
                self.save_result_img(input_img.detach().cpu().numpy(), target_img.detach().cpu().numpy(), pred_numpy,
                                     filename, os.path.join(self.save_dir, 'val_imgs'))
        del pred, concat_input, label_img
        torch.cuda.empty_cache()
        val_miou = val_miou / len(self.valloader)
        val_ious = np.round((iou_per_class / len(self.valloader)), 5).tolist()
        val_loss = val_loss / len(self.valloader)
        val_loss_kd = val_loss_kd / len(self.valloader)

        val_loss_seg = val_loss_seg / len(self.valloader)
        val_loss_list.append(val_loss)

        self.val_loss_kd_list.append(val_loss_kd)
        self.val_loss_seg_list.append(val_loss_seg)
        if val_miou >= self.best_miou:
            self.best_miou = val_miou
            self.best_miou_epoch = epoch
            self.save_checkpoint('best_val_miou.pth', self.model, epoch)

        valtxt = f"[val][epoch {epoch} mIOU: {val_miou:.4f}, IOU per class:{val_ious}---best mIOU:{self.best_miou}, best mIOU epoch: {self.best_miou_epoch}]\n"
        print(valtxt)
        if opt.save_txt:
            self.f.write(valtxt)
        if opt.save_csv:
            self.val_miou_list += [val_miou]
            self.val_loss_list += [val_loss]

    def save_checkpoint(self, filename, model, epoch):
        filename = os.path.join(self.ckpoint_path, filename)
        torch.save({'network': model.state_dict(),
                    'epoch': epoch,
                    'optimizer': self.optimizer.state_dict(),
                    'best_miou': self.best_miou, },
                   filename)

    def load_checkpoint(self, weights_path, istrain=True):
        chkpoint = torch.load(weights_path)
        self.model.load_state_dict(chkpoint['network'])
        if istrain:
            return chkpoint['epoch'], chkpoint['optimizer'], chkpoint['best_miou']

    def mask_labeling(self, y_batch: torch.Tensor, num_classes: int):
        label_pixels = list(torch.unique(y_batch, sorted=True))
        assert len(label_pixels) <= num_classes, 'too many label pixels'
        if len(label_pixels) < num_classes:
            print('label pixels error')
            label_pixels = [0, 128, 255]

        for i, px in enumerate(label_pixels):
            y_batch = torch.where(y_batch == px, i, y_batch)

        return y_batch

    def pred_to_colormap(self, pred: np.ndarray, colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])):
        pred_label = np.argmax(pred, axis=1)  # (N, H, W)
        show_pred = colormap[pred_label]  # (N, H, W, 3)
        return show_pred

    def pred_to_binary(self, pred: np.ndarray, colormap=np.array([[0., 0., 0.], [1., 1., 1.]])):
        pred_label = np.argmax(pred, axis=1)  # (N, H, W)
        show_pred = colormap[pred_label]  # (N, H, W, 2)
        return show_pred

    def save_result_img(self, input: np.ndarray, target: np.ndarray, pred: np.ndarray, filename, save_dir):
        N = input.shape[0]
        show_pred = self.pred_to_colormap(pred)
        for i in range(N):
            input_img = np.transpose(input[i], (1, 2, 0))  # (H, W, 3)
            target_img = np.transpose(np.array([target[i] / 255] * 3), (1, 2, 0))  # (3, H, W) -> (H, W, 3)
            pred_img = show_pred[i]  # (H, W, 3)
            cat_img = np.concatenate((input_img, target_img, pred_img), axis=1)  # (H, 3W, 3)
            plt.imsave(os.path.join(save_dir, filename[i]), cat_img)

    def save_result_ob_out_seg(self, input: np.ndarray, target: np.ndarray, pred: np.ndarray, ob: np.ndarray,
                               out: np.ndarray, img: np.ndarray, filename, save_dir):
        N = input.shape[0]
        show_pred = self.pred_to_colormap(pred)
        show_ob = self.pred_to_binary(ob)
        for i in range(N):
            input_img = np.transpose(input[i], (1, 2, 0))  # (H, W, 3)
            target_img = np.transpose(np.array([target[i] / 255] * 3), (1, 2, 0))  # (3, H, W) -> (H, W, 3)
            pred_img = show_pred[i]  # (H, W, 3)
            ob = show_ob[i]
            out = np.transpose(out[i], (1, 2, 0))
            img = np.transpose(img[i], (1, 2, 0))
            row1 = np.concatenate((input_img, img, target_img), axis=1)
            row2 = np.concatenate((ob, out, pred_img), axis=1)
            cat_img = np.concatenate((row1, row2), axis=0)  # (H, 6W, 3)
            plt.imsave(os.path.join(save_dir, filename[i]), cat_img)

    def save_lossgraph(self, train_loss: list, val_loss: list):
        # the graph for Loss
        epochs = list(range(0, len(train_loss)))
        plt.figure(figsize=(10, 5))
        plt.title("Loss")
        plt.plot(epochs, train_loss, label='Train loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.xlim(0, len(train_loss))
        plt.ylim(0, 2)
        plt.legend()  # 범례
        plt.savefig(os.path.join(self.save_dir, 'Loss_Graph.png'))

    def save_lossgraph2(self, train_loss_1: list, train_loss_2: list):
        # the graph for Loss
        epochs = list(range(0, len(train_loss_1)))
        plt.figure(figsize=(10, 5))
        plt.title("Training loss")
        plt.plot(train_loss_1, label='Loss kd')
        plt.plot(train_loss_2, label='Loss seg')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.xlim(0, len(train_loss_1))
        plt.ylim(0, 1)
        plt.legend()  # 범례
        plt.savefig(os.path.join(self.save_dir, 'Loss_Graph_train.png'))

    def save_lossgraph3(self, val_loss_1: list, val_loss_2: list):
        # the graph for Loss
        epochs = list(range(0, len(val_loss_1)))
        plt.figure(figsize=(10, 5))
        plt.title("Validation loss")
        plt.plot(val_loss_1, label='Loss kd')
        plt.plot(val_loss_2, label='Loss seg')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.xlim(0, len(val_loss_1))
        plt.ylim(0, 1)
        plt.legend()  # 범례
        plt.savefig(os.path.join(self.save_dir, 'Loss_Graph_val.png'))

    def save_csv(self, mode, value_list: List, filename):
        if mode == 'train':
            df = pd.DataFrame({'loss': value_list[0],
                               'lr': value_list[1],
                               'miou': value_list[2]
                               })
        if mode == 'val':
            df = pd.DataFrame({'val_loss': value_list[0],
                               'val_miou': value_list[1]})

        df.to_csv(os.path.join(os.path.abspath(self.save_dir), filename), mode='a')

    def start_timer(self):
        '''before training processes'''
        global start_time
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
        start_time = time.time()

    def end_timer_and_print(self):
        torch.cuda.synchronize()
        end_time = time.time()
        print("Total execution time = {:.3f} sec".format(end_time - start_time))
        print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kd', help='segmentation model''s name for training')
    parser.add_argument('--config', type=str, default='./config/seg_train_config.yaml', help='yaml file that has segmentation train config information')
    parser.add_argument('--save_img', type=bool, default=False, help='save result images')
    parser.add_argument('--save_txt', type=bool, default=True, help='save training process as txt file')
    parser.add_argument('--save_csv', type=bool, default=True, help='save training process as csv file')
    parser.add_argument('--save_graph', type=bool, default=True, help='save Loss graph with plt')
    parser.add_argument('--base_weight', type=float, default=1.0, help='base weight for loss calculation')
    parser.add_argument('--density_weight', type=float, default=2.0, help='weight for object density')
    parser.add_argument('--position_weight', type=float, default=1.5, help='weight for position-based weighting')
    parser.add_argument('--density_kernel_size', type=int, default=15, help='kernel size for density calculation')
    parser.add_argument('--position_decay_factor', type=float, default=0.8, help='decay factor for position weights')
    parser.add_argument('--weight_T', type=str, default='D:/save/2nd_paper/cwfid/DDOS_Net-ep400-train-1-0/ckpoints/best_val_miou.pth', help='teacher weight')
    opt = parser.parse_args()

    if opt.model == 'kd':
        model_T = models.DDOS_Net(in_channels=4, num_classes=3, pretrained=True)
        model = models.SDS_Net(in_channels=4, num_classes=3, first_outchannels=32)



    with open(opt.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    trainer = Trainer(opt, cfg, model_T, model)
    trainer.train(opt)
