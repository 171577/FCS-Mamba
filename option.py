import argparse
import torch
def str2bool(v):
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    if v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    
    def init(self):
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='LEVIR', help='project name')
        self.parser.add_argument('--dataroot', type=str, default="/root/lanyun-tmp/SWCD-main/SWCD-main/data/LEVIR")
        self.parser.add_argument('--dataset', type=str, default='LEVIR')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints391', help='models are saved here')
        self.parser.add_argument('--weight_dir', type=str,
                                 default="./checkpoints160/LEVIR/best_model.pth", help='models are saved here')
        self.parser.add_argument('--label_rate', type=str, default='30', help='10, 30, 50, None')
        self.parser.add_argument('--result_dir', type=str, default='./results311', help='results are saved here')
        self.parser.add_argument('--load_pretrain', type=str2bool, default=True)
        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--input_size', type=int, default=256)
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--num_epochs', type=int, default=1000)
        self.parser.add_argument('--warmup_epochs', type=int, default=100)
        self.parser.add_argument('--num_workers', type=int, default=8, help='#threads for loading data')
        self.parser.add_argument('--lr', type=float, default=8e-4)
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)
        
        # ========== 损失函数权重参数 ==========
        self.parser.add_argument('--use_ce_loss', type=str2bool, default=True,
                                help='[ABLATION] enable cross entropy loss for main/aux branches')

        self.parser.add_argument('--use_dice_loss', type=str2bool, default=True,
                                help='[ABLATION] enable dice loss for main/aux branches')
        self.parser.add_argument('--use_edge_loss', type=str2bool, default=True,
                                help='[ABLATION] enable edge loss')
        self.parser.add_argument('--use_sim_loss', type=str2bool, default=True,
                                help='[ABLATION] enable similarity/weak supervision loss')
        self.parser.add_argument('--lambda_sim', type=float, default=0.5,
                                help='weight for similarity/weak supervision loss')
        self.parser.add_argument('--lambda_dice', type=float, default=0.9,
                                help='weight for dice loss')
        self.parser.add_argument('--lambda_edge', type=float, default=0.9,
                                help='weight for edge loss')
        self.parser.add_argument('--ce_weight0', type=float, default=1.0,
                                help='cross entropy weight for class 0')
        self.parser.add_argument('--ce_weight1', type=float, default=2.0,
                                help='cross entropy weight for class 1')
        self.parser.add_argument('--label_smoothing', type=float, default=0.05,
                                help='label smoothing factor for cross entropy loss')
        self.parser.add_argument('--eta_min', type=float, default=1e-7,
                                help='minimum learning rate for scheduler')

        self.parser.add_argument('--use_depthwise', type=str2bool, default=True)
        self.parser.add_argument('--use_feedback_decoder', type=str2bool, default=True)

        self.parser.add_argument('--use_cross_scale_attn', type=str2bool, default=True,
                                help='enable cross-scale attention')
        self.parser.add_argument('--use_feature_align', type=str2bool, default=True,
                                help='enable feature alignment')
        self.parser.add_argument('--use_weak_feature_align', type=str2bool, default=True,
                                help='enable feature alignment for weak supervision branch (default: follow use_feature_align)')
        self.parser.add_argument('--use_weak_supervision', type=str2bool, default=True,
                                help='enable weak supervision / similarity branch')
        self.parser.add_argument('--use_weak_label', type=str2bool, default=True,
                                help='use weak label instead of downsampled strong label for weak supervision loss')
        
        # ========== 消融实验 2, 3, 4, 5 开关 ==========
        self.parser.add_argument('--use_fem', type=str2bool, default=False,
                                help='[ABLATION 2] enable Feature Enhancement Module in weak branch')
        self.parser.add_argument('--use_im', type=str2bool, default=False,
                                help='[ABLATION 3] enable Interaction Module in weak branch')
        self.parser.add_argument('--use_weak_guidance', type=str2bool, default=True,
                                help='[ABLATION 4] enable weak similarity map guidance for main branch')
        # 注: --use_weak_label (已存在) 对应消融实验 5: True=真实弱标签, False=强标签下采样
        self.parser.add_argument('--use_weak_label_for_main', type=str2bool, default=False,
                                help='[ABLATION] use weak label for main branch supervision instead of strong label')
        self.parser.add_argument('--weak_main_weight', type=float, default=1.0,
                                help='[ABLATION] weight for weak label loss in main branch (0-1, default=1.0)')
        self.parser.add_argument('--use_mixed_supervision', type=str2bool, default=False,
                                help='[ABLATION] use mixed strong+weak labels for main branch supervision')
        self.parser.add_argument('--mixed_alpha', type=float, default=0.5,
                                help='[ABLATION] weight for strong label in mixed supervision (0-1, default=0.5). '
                                     'Loss = alpha * loss_strong + (1-alpha) * loss_weak')
        
        self.parser.add_argument('--use_mdem', type=str2bool, default=True,
                                help='enable MDEM module')
        self.parser.add_argument('--use_sg_delta', type=str2bool, default=True,
                    help='[ABLATION] enable Similarity-Guided Delta modulation inside Mamba blocks')
        self.parser.add_argument('--use_mask_scan', type=str2bool, default=True,
                    help='[ABLATION] enable Mask-Prompted scan routing in Mamba blocks')
        self.parser.add_argument('--use_cross_gate', type=str2bool, default=True,
                    help='[ABLATION] enable Cross-Temporal state gating in Mamba blocks')
        self.parser.add_argument('--use_combined_mamba', type=str2bool, default=True,
                    help='[ABLATION] one-click enable for use_sg_delta/use_mask_scan/use_cross_gate')
        self.parser.add_argument('--main_branch_type', type=str, default='mamba')
                                
        # ========== GSRA 几何-语义校正注意力开关 ==========
        self.parser.add_argument('--use_gsra', type=str2bool, default=False,
                help='[ABLATION] enable Geometric-Semantic Rectification Attention (GSRA)')
        self.parser.add_argument('--gsra_token_projection', type=str, default='concat',
                help='GSRA projection mode: base or concat')
        self.parser.add_argument('--gsra_window_size', type=int, default=4,
                help='window size for GSRA local attention')
        self.parser.add_argument('--gsra_num_heads', type=int, default=2,
                help='number of attention heads for GSRA')
        self.parser.add_argument('--gsra_geo_dim', type=int, default=3,
                help='input geometric feature dimension for GSRA')
        self.parser.add_argument('--gsra_dino_dim', type=int, default=256,
                help='semantic feature dimension used by GSRA')
        
        self.parser.add_argument('--backbone_type', type=str, default='resnet18')
        self.parser.add_argument('--weak_feature_source', type=str, default='resnet18')
        self.parser.add_argument('--resnet_pretrained', type=str2bool, default=True)

        self.parser.add_argument('--vssm_pretrained', type=str, default='/root/lanyun-tmp/MSDANet-main_new/MSDANet/model/vssm1_tiny_0230s_ckpt_epoch_264.pth')
        self.parser.add_argument('--vssm_patch_size', type=int, default=4)
        self.parser.add_argument('--vssm_depths', type=str, default='2,2,4,2')
        self.parser.add_argument('--vssm_dims', type=str, default='96')
        self.parser.add_argument('--vssm_norm_layer', type=str, default='ln2d')
        self.parser.add_argument('--vssm_forward_type', type=str, default='v3noz')
        self.parser.add_argument('--vssm_ssm_d_state', type=int, default=16)
        self.parser.add_argument('--vssm_ssm_ratio', type=float, default=2.0)
        self.parser.add_argument('--vssm_ssm_dt_rank', type=str, default='auto')
        self.parser.add_argument('--vssm_ssm_conv', type=int, default=3)
        self.parser.add_argument('--vssm_ssm_conv_bias', type=str2bool, default=True)
        self.parser.add_argument('--vssm_ssm_drop_rate', type=float, default=0.0)
        self.parser.add_argument('--vssm_ssm_init', type=str, default='v0')
        self.parser.add_argument('--vssm_mlp_ratio', type=float, default=4.0)
        self.parser.add_argument('--vssm_mlp_drop_rate', type=float, default=0.0)

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

                     
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        if isinstance(getattr(self.opt, 'vssm_pretrained', None), str) and self.opt.vssm_pretrained.strip() == '':
            self.opt.vssm_pretrained = None

        if isinstance(getattr(self.opt, 'vssm_depths', None), str):
            depths_str = self.opt.vssm_depths.strip()
            if depths_str:
                self.opt.vssm_depths = tuple(int(x.strip()) for x in depths_str.split(',') if x.strip() != '')

        if isinstance(getattr(self.opt, 'vssm_dims', None), str):
            dims_str = self.opt.vssm_dims.strip()
            if ',' in dims_str:
                self.opt.vssm_dims = [int(x.strip()) for x in dims_str.split(',') if x.strip() != '']
            elif dims_str:
                self.opt.vssm_dims = int(dims_str)

        if bool(getattr(self.opt, 'use_combined_mamba', False)):
            self.opt.use_sg_delta = True
            self.opt.use_mask_scan = True
            self.opt.use_cross_gate = True

        args = vars(self.opt)

        return self.opt
