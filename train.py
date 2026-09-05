import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from option import Options
from data.cd_dataset import DataLoader
from model.module.MSDANet import MSDANet
from model.loss.dice import dice_loss_v1
from model.loss.edge_loss import Sobel
from tqdm import tqdm
from util.metric_tool import ConfuseMatrixMeter
import os
import numpy as np
import random
import logging
import datetime
import json
logging.getLogger('PIL').setLevel(logging.WARNING)

def _optimizer_to(optim: torch.optim.Optimizer, device: torch.device):
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)

def init_logging(filedir: str):
                                                
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')
    logger = logging.getLogger()
    fh = logging.FileHandler(filename=filedir + '/log_' + get_date_str() + '.txt')
    sh = logging.StreamHandler()
    formatter_fh = logging.Formatter('%(asctime)s %(message)s')
    formatter_sh = logging.Formatter('%(message)s')
    fh.setFormatter(formatter_fh)
    sh.setFormatter(formatter_sh)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(10)
    fh.setLevel(10)
    sh.setLevel(10)
    return logging

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.num_epochs = getattr(opt, 'num_epochs', 500)
        self.start_epoch = 0

        train_loader = DataLoader(opt)
        self.train_data = train_loader.load_data()

        opt.phase = 'val'
        opt.batch_size = 64
        val_loader = DataLoader(opt)
        self.val_data = val_loader.load_data()

        opt.phase = 'train'
        opt.batch_size = getattr(opt, 'batch_size', 8)

        self.model = MSDANet.from_opt(
            opt,
            input_nc=3,
            output_nc=2,
            use_depthwise=True,
            use_feedback_decoder=True,
        ).cuda()

        class_weights = torch.tensor([
            float(getattr(opt, 'ce_weight0', 1.0)),
            float(getattr(opt, 'ce_weight1', 2.0)),
        ]).cuda()
        self.criterion_ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(getattr(opt, 'label_smoothing', 0.05))
        )
        self.criterion_dice = dice_loss_v1()
        self.criterion_edge = nn.BCEWithLogitsLoss()
        self.sobel = Sobel().cuda()

        self.lambda_sim = getattr(opt, 'lambda_sim', 0.1)
        self.lambda_dice = getattr(opt, 'lambda_dice', 0.5)
        self.lambda_edge = getattr(opt, 'lambda_edge', 0.5)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=getattr(opt, 'lr', 3e-4),
            weight_decay=getattr(opt, 'weight_decay', 5e-4)
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=float(getattr(opt, 'eta_min', 1e-7))
        )

        self.running_metric = ConfuseMatrixMeter(n_class=2)
        self.best_epoch = 0
        self.best_f1 = -1.0

        resume_path = getattr(opt, 'resume', None)
        if isinstance(resume_path, str) and resume_path.strip() != '':
            self._load_checkpoint(resume_path.strip())

        self._log_model_switches()
        
        # 打印弱标签配置信息
        logging.info("=" * 80)
        logging.info("弱标签配置:")
        logging.info("损失函数消融开关:")
        logging.info(f"  - use_ce_loss: {getattr(opt, 'use_ce_loss', True)}")
        logging.info(f"  - use_dice_loss: {getattr(opt, 'use_dice_loss', True)}")
        logging.info(f"  - use_edge_loss: {getattr(opt, 'use_edge_loss', True)}")
        logging.info(f"  - use_sim_loss: {getattr(opt, 'use_sim_loss', True)}")
        logging.info(f"  - use_weak_supervision: {getattr(opt, 'use_weak_supervision', False)}")
        logging.info(f"  - use_weak_label (for sim branch): {getattr(opt, 'use_weak_label', False)}")
        logging.info(f"  - use_weak_label_for_main: {getattr(opt, 'use_weak_label_for_main', False)}")
        logging.info(f"  - use_mixed_supervision: {getattr(opt, 'use_mixed_supervision', False)}")
        if getattr(opt, 'use_mixed_supervision', False):
            logging.info(f"  - mixed_alpha: {getattr(opt, 'mixed_alpha', 0.5)}")
        logging.info(f"  - lambda_sim: {self.lambda_sim}")
        logging.info(f"  - lambda_dice: {self.lambda_dice}")
        logging.info(f"  - lambda_edge: {self.lambda_edge}")
        logging.info("=" * 80)

    def _log_model_switches(self):
        # Print a compact and complete summary of architecture/ablation switches.
        logging.info("=" * 80)
        logging.info("模型开关总览:")

        model_groups = {
            'core_model': [
                'backbone_type', 'main_branch_type', 'weak_feature_source',
                'use_depthwise', 'use_feedback_decoder', 'resnet_pretrained'
            ],
            'feature_alignment': [
                'use_feature_align', 'use_weak_feature_align'
            ],
            'cross_scale': [
                'use_cross_scale_attn'
            ],
            'ablation_modules': [
                'use_fem', 'use_im',
                'use_weak_guidance', 'use_mdem'
            ],
            'gsra': [
                'use_gsra', 'gsra_token_projection', 'gsra_window_size',
                'gsra_num_heads', 'gsra_geo_dim', 'gsra_dino_dim'
            ],
            'vssm': [
                'vssm_pretrained', 'vssm_patch_size', 'vssm_depths', 'vssm_dims',
                'vssm_norm_layer', 'vssm_forward_type', 'vssm_ssm_d_state',
                'vssm_ssm_ratio', 'vssm_ssm_dt_rank', 'vssm_ssm_conv',
                'vssm_ssm_conv_bias', 'vssm_ssm_drop_rate', 'vssm_ssm_init',
                'vssm_mlp_ratio', 'vssm_mlp_drop_rate'
            ]
        }

        for group_name, keys in model_groups.items():
            logging.info(f"[{group_name}]")
            for key in keys:
                if hasattr(self.opt, key):
                    logging.info(f"  - {key}: {getattr(self.opt, key)}")

        known_keys = set()
        for keys in model_groups.values():
            known_keys.update(keys)

        extra_bool_keys = []
        for key, value in sorted(vars(self.opt).items()):
            if key in known_keys:
                continue
            if isinstance(value, bool) and (
                key.startswith('use_')
                or key.startswith('cross_scale_')
                or key.startswith('mamba_')
                or key.startswith('gsra_')
            ):
                extra_bool_keys.append(key)

        if extra_bool_keys:
            logging.info("[other_boolean_switches]")
            for key in extra_bool_keys:
                logging.info(f"  - {key}: {getattr(self.opt, key)}")

        logging.info("=" * 80)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = MSDANet.remap_legacy_state_dict_keys(ckpt['model_state_dict'])
            self.model.load_state_dict(state_dict, strict=True)
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                _optimizer_to(self.optimizer, device=next(self.model.parameters()).device)
            if 'scheduler_state_dict' in ckpt:
                try:
                    self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception:
                    pass
            if 'epoch' in ckpt:
                self.start_epoch = int(ckpt['epoch'])
            if 'best_f1' in ckpt:
                self.best_f1 = float(ckpt['best_f1'])
            if 'best_epoch' in ckpt:
                self.best_epoch = int(ckpt['best_epoch'])
        else:
            state_dict = MSDANet.remap_legacy_state_dict_keys(ckpt)
            self.model.load_state_dict(state_dict, strict=True)

    def _check_label_range(self, label: torch.Tensor, num_classes: int, name: str = "label"):
        if label.dtype != torch.long:
            raise RuntimeError(f"{name} must be torch.long for CE loss, got dtype={label.dtype}")
        if label.dim() != 3:
            raise RuntimeError(f"{name} must be [B,H,W], got shape={tuple(label.shape)}")
        if not torch.isfinite(label).all():
            raise RuntimeError(f"{name} contains non-finite values")

        min_v = int(label.min().item())
        max_v = int(label.max().item())
        if min_v < 0 or max_v >= int(num_classes):
            raise RuntimeError(
                f"{name} out of range for num_classes={int(num_classes)}: min={min_v}, max={max_v}"
            )

    def _tensor_stats(self, x: torch.Tensor, name: str):
        if x is None:
            return f"{name}=None"
        if not torch.is_tensor(x):
            return f"{name} type={type(x)}"
        with torch.no_grad():
            finite = torch.isfinite(x)
            finite_ratio = float(finite.float().mean().item()) if x.numel() > 0 else 1.0
            if finite.any():
                x_f = x[finite]
                mn = float(x_f.min().item())
                mx = float(x_f.max().item())
            else:
                mn, mx = float('nan'), float('nan')
        return f"{name} shape={tuple(x.shape)} finite={finite_ratio:.4f} min={mn:.6g} max={mx:.6g}"

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_data, desc=f"Epoch {epoch + 1}/{self.num_epochs} [TRAIN]")

        total_loss = 0.0
        valid_steps = 0
        for batch in pbar:
            img1 = batch['img1'].cuda()
            img2 = batch['img2'].cuda()
            label = batch['label'].cuda().long()
            use_ce_loss = bool(getattr(self.opt, 'use_ce_loss', True))
            use_dice_loss = bool(getattr(self.opt, 'use_dice_loss', True))
            use_edge_loss = bool(getattr(self.opt, 'use_edge_loss', True))
            use_sim_loss = bool(getattr(self.opt, 'use_sim_loss', True))
            use_weak_supervision = bool(getattr(self.opt, 'use_weak_supervision', False))
            use_weak_for_main = bool(getattr(self.opt, 'use_weak_label_for_main', False))
            use_mixed = bool(getattr(self.opt, 'use_mixed_supervision', False))
            use_weak_label_for_sim = bool(getattr(self.opt, 'use_weak_label', False))

            need_label_weak = (use_weak_for_main or use_mixed) or (use_weak_supervision and use_weak_label_for_sim)
            if need_label_weak:
                if 'label_weak' not in batch:
                    raise KeyError(
                        "batch is missing 'label_weak' but current config requires weak labels. "
                        "Disable weak-label usage or provide label_weak in the dataset."
                    )
                label_weak = batch['label_weak'].cuda().long()
            else:
                label_weak = None
            name = batch.get('name', None)

            self.optimizer.zero_grad()
            output = self.model(img1, img2, gt_mask=label)
            pred = output[0]

            if not (use_ce_loss or use_dice_loss or use_edge_loss or (use_sim_loss and use_weak_supervision)):
                raise RuntimeError(
                    "All loss components are disabled (use_ce_loss/use_dice_loss/use_edge_loss/use_sim_loss). "
                    "At least one loss must be enabled to train. "
                    "Tip: set --use_ce_loss true (or enable any other loss)."
                )

            if not torch.isfinite(pred).all():
                logging.error(
                    f"Non-finite pred encountered; skip step. epoch={epoch} name={name}; "
                    + self._tensor_stats(pred, 'pred')
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self._check_label_range(label, num_classes=int(pred.shape[1]), name="label")

            sim_fused = output[2] if len(output) > 2 and torch.is_tensor(output[2]) else None

            # ========== 主分支损失计算 ==========
            # 根据配置选择使用强标签、弱标签或混合监督
            use_weak_for_main = use_weak_for_main
            use_mixed = use_mixed
            
            if use_weak_for_main:
                if label_weak is None:
                    raise RuntimeError("label_weak is required for main-branch weak supervision, but it is None")
                if use_mixed:
                    # 混合监督: alpha * 强标签 + (1-alpha) * 弱标签
                    alpha = getattr(self.opt, 'mixed_alpha', 0.5)
                    loss_ce_strong = self.criterion_ce(pred, label) if use_ce_loss else torch.tensor(0.0, device=label.device)
                    loss_ce_weak = self.criterion_ce(pred, label_weak) if use_ce_loss else torch.tensor(0.0, device=label.device)
                    loss_ce = alpha * loss_ce_strong + (1 - alpha) * loss_ce_weak
                    
                    loss_dice_strong = self.criterion_dice(pred, label.unsqueeze(1).float()) if use_dice_loss else torch.tensor(0.0, device=label.device)
                    loss_dice_weak = self.criterion_dice(pred, label_weak.unsqueeze(1).float()) if use_dice_loss else torch.tensor(0.0, device=label.device)
                    loss_dice = alpha * loss_dice_strong + (1 - alpha) * loss_dice_weak
                else:
                    # 纯弱标签监督
                    loss_ce = self.criterion_ce(pred, label_weak) if use_ce_loss else torch.tensor(0.0, device=label.device)
                    loss_dice = self.criterion_dice(pred, label_weak.unsqueeze(1).float()) if use_dice_loss else torch.tensor(0.0, device=label.device)
            else:
                # 原始强标签监督
                loss_ce = self.criterion_ce(pred, label) if use_ce_loss else torch.tensor(0.0, device=label.device)
                loss_dice = self.criterion_dice(pred, label.unsqueeze(1).float()) if use_dice_loss else torch.tensor(0.0, device=label.device)

            # Ensure the zero-loss placeholders are connected to the computation graph.
            # This prevents `loss.backward()` from erroring when some components are disabled.
            if not use_ce_loss:
                loss_ce = pred.sum() * 0.0
            if not use_dice_loss:
                loss_dice = pred.sum() * 0.0
            
            loss = loss_ce + self.lambda_dice * loss_dice

            # ========== 边缘损失计算 ==========
            loss_edge = torch.tensor(0.0, device=label.device)

            if use_edge_loss and self.lambda_edge > 0:
                # Convert multi-channel prediction to single channel for edge detection
                # Use softmax to get probabilities, then take the positive class (channel 1)
                pred_single = F.softmax(pred, dim=1)[:, 1:2, :, :]  # [B, 1, H, W]
                
                # 根据配置选择使用强标签或弱标签
                if use_weak_for_main:
                    if label_weak is None:
                        raise RuntimeError("label_weak is required for edge loss weak supervision, but it is None")
                    if use_mixed:
                        label_single_strong = label.unsqueeze(1).float()
                        label_single_weak = label_weak.unsqueeze(1).float()
                        loss_edge_strong = self.criterion_edge(self.sobel(pred_single), self.sobel(label_single_strong))
                        loss_edge_weak = self.criterion_edge(self.sobel(pred_single), self.sobel(label_single_weak))
                        loss_edge = alpha * loss_edge_strong + (1 - alpha) * loss_edge_weak
                    else:
                        label_single = label_weak.unsqueeze(1).float()
                        loss_edge = self.criterion_edge(self.sobel(pred_single), self.sobel(label_single))
                else:
                    label_single = label.unsqueeze(1).float()
                    loss_edge = self.criterion_edge(self.sobel(pred_single), self.sobel(label_single))
            else:
                loss_edge = pred.sum() * 0.0

            loss_sim = torch.tensor(0.0, device=label.device)
            if use_sim_loss and use_weak_supervision and sim_fused is not None:
                # 根据配置选择使用弱标签还是强标签的下采样版本
                if use_weak_label_for_sim:
                    # 使用真正的弱标签
                    if label_weak is None:
                        raise RuntimeError("label_weak is required for sim-branch weak supervision, but it is None")
                    label_for_weak = label_weak.float().unsqueeze(1)
                    label_weak_target = F.adaptive_max_pool2d(label_for_weak, sim_fused.shape[2:])
                else:
                    # 使用强标签的下采样版本（原始方法）
                    label_change = label.float().unsqueeze(1)
                    label_weak_target = F.adaptive_max_pool2d(label_change, sim_fused.shape[2:])
                
                sim_fused = torch.nan_to_num(sim_fused, nan=0.5, posinf=1.0, neginf=0.0)
                if not torch.isfinite(sim_fused).all():
                    raise RuntimeError(
                        f"sim_fused contains non-finite values before BCE: "
                        f"min={float(sim_fused.min().item())}, max={float(sim_fused.max().item())}"
                    )
                sim_fused = sim_fused.clamp(min=1e-6, max=1.0 - 1e-6)
                loss_sim = F.binary_cross_entropy(sim_fused, label_weak_target)
            else:
                loss_sim = pred.sum() * 0.0

            loss = loss + self.lambda_sim * loss_sim + self.lambda_edge * loss_edge

            if not torch.isfinite(loss).all():
                logging.error(
                    f"Non-finite loss encountered; skip step. epoch={epoch} name={name}; "
                    f"loss={float(loss.detach().item()) if loss.numel()==1 else 'tensor'}; "
                    f"loss_ce={float(loss_ce.detach().item()) if torch.is_tensor(loss_ce) and loss_ce.numel()==1 else 'tensor'}; "
                    f"loss_dice={float(loss_dice.detach().item()) if torch.is_tensor(loss_dice) and loss_dice.numel()==1 else 'tensor'}; "
                    f"loss_sim={float(loss_sim.detach().item()) if torch.is_tensor(loss_sim) and loss_sim.numel()==1 else 'tensor'}; "
                    + self._tensor_stats(pred, 'pred') + "; "
                    + self._tensor_stats(sim_fused, 'sim_fused')
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                logging.error(
                    f"Non-finite grad_norm encountered; skip step. epoch={epoch} name={name}; grad_norm={grad_norm}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.optimizer.step()

            total_loss += loss.item()
            valid_steps += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        self.scheduler.step(epoch + 1)

        epoch_len = max(1, valid_steps)
        avg_loss = total_loss / epoch_len
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        self.running_metric.clear()

        pbar = tqdm(self.val_data, desc=f"Epoch {epoch + 1}/{self.num_epochs} [VAL]")
        with torch.no_grad():
            for batch in pbar:
                img1 = batch['img1'].cuda()
                img2 = batch['img2'].cuda()
                label = batch['label']

                pred, _ = self.model(img1, img2)
                pred = torch.argmax(pred, dim=1)

                self.running_metric.update_cm(pr=pred.cpu().numpy(), gt=label.cpu().numpy())

        scores = self.running_metric.get_scores()
        macro_f1 = (scores.get('F1_0', 0) + scores.get('F1_1', 0)) / 2

        val_metrics = {
            'acc': scores.get('acc', 0),
            'miou': scores.get('miou', 0),
            'F1_1': scores.get('F1_1', 0),
            'precision_1': scores.get('precision_1', 0),
            'recall_1': scores.get('recall_1', 0),
            'macro_f1': macro_f1,
        }

        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1
            self.best_epoch = epoch + 1
            checkpoint = {
                'epoch': self.best_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_f1': self.best_f1,
                'metrics': scores,
            }
            checkpoint_path = os.path.join(self.opt.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)

        return val_metrics

    def run(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            logging.info(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_acc={val_metrics['acc']:.4f} | "
                f"val_miou={val_metrics['miou']:.4f} | "
                f"val_F1_1={val_metrics['F1_1']:.4f} | "
                f"val_precision_1={val_metrics['precision_1']:.4f} | "
                f"val_recall_1={val_metrics['recall_1']:.4f}"
            )

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_f1': self.best_f1,
                'best_epoch': self.best_epoch,
            }
            checkpoint_path = os.path.join(self.opt.checkpoint_dir, 'last_model.pth')
            torch.save(checkpoint, checkpoint_path)
def main():
    opt = Options().parse()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    init_logging(opt.checkpoint_dir)
    setup_seed(1)
    trainer = Trainer(opt)
    trainer.run()

if __name__ == '__main__':
    main()
