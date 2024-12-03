# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# This file contains the loss functions for the audio reconstruction task

import utils.utils as utils
import models.utils as model_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute_loss(self, *args, **kwargs):
        return torch.tensor(0)
    def forward(self, *args, **kwargs):
        return torch.tensor(0)


class AudioReconLossMoE(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_loss(self, pred_vals, gt_vals, q, *args, **kwargs):
        recon_loss = (((pred_vals - gt_vals.unsqueeze(-1)) ** 2)*q.unsqueeze(-2)).mean()
        return recon_loss


class AudioReconLossSingle(nn.Module):
    def __init__(self, metric='mean'):
        super().__init__()
        self.metric = metric

    def compute_loss(self, pred_vals, gt_vals, *args, **kwargs):
        if self.metric == 'mean':
            recon_loss = ((pred_vals - gt_vals) ** 2).mean()
        elif self.metric == 'median':
            recon_loss = ((pred_vals - gt_vals) ** 2).median()
        else:
            raise ValueError('Invalid metric for AudioReconLossSingle')
        return recon_loss



class AudioReconLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': AudioReconLossSingle(),
                     'moe': AudioReconLossMoE(),
                     'sparsemoe': AudioReconLossMoE()}
        self.loss = loss_dict[loss_type]

    def forward(self, pred_vals, gt_vals, q, *args, **kwargs):
        return self.loss.compute_loss(pred_vals, gt_vals, q)

class MSEEachExpert(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_vals, gt_vals, *args, **kwargs):
        mse = ((pred_vals - gt_vals.unsqueeze(-1)) ** 2).mean(1).mean(1).squeeze()
        return mse

###################################### Balancing Loss #############################################

class BalancingLossMoE(nn.Module):
    def __init__(self, n_experts, sample_bias_correction):
        super().__init__()
        self.n_experts = n_experts
        self.sample_bias_correction = sample_bias_correction

    def top1(self, t):
        values, index = t.topk(k=1, dim=-1)
        values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
        return values, index

    def compute_loss(self, q, *args, **kwargs):
        _, index_1 = self.top1(q)
        mask_1 = F.one_hot(index_1, self.n_experts).float()
        density_1_proxy = q.mean(dim=-2)
        density_1 = mask_1.mean(dim=-2)
        balancing_loss = (density_1_proxy * density_1).mean() * float(self.n_experts ** 2)
        return balancing_loss

class BalancingLoss(nn.Module):
    def __init__(self, loss_type, n_experts, sample_bias_correction):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': DummyLoss(),
                     'moe': BalancingLossMoE(n_experts, sample_bias_correction),
                     'sparsemoe': DummyLoss()}
        self.loss = loss_dict[loss_type]

    def forward(self, q, *args, **kwargs):
        return self.loss.compute_loss(q)

###################################### Segmentation Loss #############################################

class SegmentationLossMoE(nn.Module):
    def __init__(self, n_experts, seg_type='ce'):
        super().__init__()
        self.n_experts = n_experts
        self.seg_type = seg_type

    def compute_loss(self, q, gt_segment, *args, **kwargs):
        segmentation_loss = F.cross_entropy(q, gt_segment.long(), label_smoothing=0.01)
        return segmentation_loss


class SegmentationLoss(nn.Module):
    def __init__(self, loss_type, n_experts, seg_type='ce'):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': DummyLoss(),
                     'moe': SegmentationLossMoE(n_experts, seg_type),
                     'sparsemoe': SegmentationLossMoE(n_experts, seg_type)}
        self.loss = loss_dict[loss_type]

    def forward(self, q, gt_segment, *args, **kwargs):
        return self.loss.compute_loss(q, gt_segment)

###################################### imporance and load Loss #############################################
class LoadLossSparseMoE(nn.Module):
    def __init__(self):
        super().__init__()

    def cv_squared(self, x, eps=1e-5):
        return x.float().var() / (x.float().abs().mean() + eps) #(x.float().mean()**2 + eps), the squared version is from the original code but it is not correct

    def compute_loss(self, importance, load, q, *args, **kwargs):
        loss = self.cv_squared(importance) + self.cv_squared(load)
        return loss

class LoadLossMoE(nn.Module):
    def __init__(self):
        super().__init__()

    def cv_squared(self, x, eps=1e-5):
        return x.float().var() / (x.float().abs().mean() + eps) #(x.float().mean()**2 + eps), the squared version is from the original code but it is not correct

    def compute_loss(self, importance, load, q, *args, **kwargs):
        importance = q.sum(1)
        load = (q > 0).sum(1)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        return loss

class LoadLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': DummyLoss(),
                     'moe':  LoadLossMoE(),
                     'sparsemoe':  LoadLossSparseMoE()}

        self.loss = loss_dict[loss_type]
    def forward(self, importance, load, q, *args, **kwargs):
        return self.loss.compute_loss(importance, load, q)


###################################### audio reconstruction error loss applied to all experts without q Loss #############################################
class AudioReconAllLossMoE(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_loss(self, pred_vals, gt_vals, *args, **kwargs):
        recon_loss = (((pred_vals - gt_vals.unsqueeze(-1)) ** 2)).mean()
        return recon_loss

class AudioReconAllLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': AudioReconLossSingle(),
                     'moe': AudioReconAllLossMoE(),
                     'sparsemoe': AudioReconAllLossMoE()}
        self.loss = loss_dict[loss_type]

    def forward(self, pred_vals, gt_vals, *args, **kwargs):
        return self.loss.compute_loss(pred_vals, gt_vals)


LOSS_LIST = ['audiorecon', 'audioreconall', 'balance', 'segmentation',  'load', 'init'] #'eikonal', 'lipschitz',
class AudioLoss(nn.Module):
    def __init__(self, cfg, model_name, in_dim=2, model=None, n_experts=1):
        super().__init__()
        self.sample_bias_correction = cfg['sample_bias_correction']
        if '_moe' in model_name:
            self.model_type = 'moe'
            moe_indicator = True
        elif 'sparsemoe' in model_name:
            self.model_type = 'sparsemoe'
            moe_indicator = False
        elif 'sparseidfmoe' in model_name:
            self.model_type = 'sparsemoe'
            moe_indicator = False
        else:
            self.model_type = 'single'
            moe_indicator = False

        self.model = model
        self.gradient_comp = utils.experts_gradient if moe_indicator else utils.gradient
        self.weight_dict = {}

        required_loss_list, weights = model_utils.parse_loss_string(cfg['loss_type'])
        self.required_loss_dict, self.weight_dict = model_utils.build_loss_dictionary(required_loss_list, weights,
                                                                                      self.model_type, full_loss_list=LOSS_LIST)

        self.audiorecon_loss = AudioReconLoss(self.required_loss_dict['audiorecon'])
        self.audioreconall_loss = AudioReconAllLoss(self.required_loss_dict['audioreconall'])
        self.balance_loss = BalancingLoss(self.required_loss_dict['balance'], n_experts, self.sample_bias_correction)
        self.segmentation_loss = SegmentationLoss(self.required_loss_dict['segmentation'], n_experts, cfg['segmentation_type'])
        self.init_signal_loss = AudioReconAllLoss(self.required_loss_dict['init'])
        self.load_loss = LoadLoss(self.required_loss_dict['load'])
        self.q_entropy_metric_str = cfg['entropy_metric']


        self.recon_mse = AudioReconLossSingle()
        self.mse_each_expert = MSEEachExpert()


    def forward(self, output_pred, coords, gt={'gt_data': None, 'segments': None, 'aux': None, 'init_signal': None}, model=None):
        q = None
        if self.model_type == 'moe' or self.model_type == 'sparsemoe' or self.model_type == 'sparseidfmoe':
            q = output_pred['nonmnfld_q'].permute(0, 2, 1)
            raw_q = output_pred['nonmnfld_raw_q'].permute(0, 2, 1).squeeze()
            final_audio = output_pred['selected_nonmanifold_pnts_pred']
            gt_segments = gt['segment'].squeeze()
            if self.model_type == 'sparsemoe':
                nonmnfld_selected_expert_idx = output_pred['nonmnfld_selected_expert_idx']
                q = torch.gather(q, dim=-1, index=nonmnfld_selected_expert_idx[None, None, :])# (1, n_nm)
        else:
            final_audio = output_pred['nonmanifold_pnts_pred'].permute(0, 2, 1)
            raw_q, gt_segments = None, None

        if output_pred['nonmanifold_pnts_pred'].dim() == 4:
            pred_audio = output_pred['nonmanifold_pnts_pred'].permute(0, 2, 3, 1)
        else:
            pred_audio = output_pred['nonmanifold_pnts_pred'].permute(0, 2, 1)

        if self.sample_bias_correction: # scale q by the number of pixel it "sees" so computing the mean is co
            q = q * q.shape[1] / torch.clamp(q.sum(-2, keepdim=True), 0.00001)

        audiorecon_term = self.audiorecon_loss(pred_audio, gt['gt_data'], q)
        audioreconall_term = self.audioreconall_loss(pred_audio, gt['gt_data'])
        balance_term = self.balance_loss(q)
        segmentation_term = self.segmentation_loss(raw_q, gt_segments)

        if 'init_signal' in gt and gt['init_signal'] is not None:
            init_term = self.init_signal_loss(final_audio, gt['init_signal'])
        else:
            init_term = torch.tensor(0)

        load_term = self.load_loss(output_pred.get('importance', None), output_pred.get('load', None), q)

        loss = (audiorecon_term * self.weight_dict['audiorecon'] + balance_term * self.weight_dict['balance'] +
                segmentation_term * self.weight_dict['segmentation'] +
                # eikonal_term * self.weight_dict['eikonal'] + lipschitz_term * self.weight_dict['lipschitz'] +
                load_term * self.weight_dict['load'] + audioreconall_term * self.weight_dict['audioreconall'] +
                init_term * self.weight_dict['init'])

        # compute error measures
        with torch.no_grad():
            recon_error = self.recon_mse.compute_loss(final_audio.squeeze(), gt['gt_data'].squeeze())
            # psnr = self.psnr(gt['gt_data'], recon_error)


        out = {'loss': loss, 'audiorecon_term': audiorecon_term, 'balance_term': balance_term,
                'reconerror_term': recon_error,  'segmentation_term': segmentation_term,
               'load_term': load_term,  'audioreconall_term': audioreconall_term,
               'init_term': init_term}

        if not self.model_type == 'single' and not self.model_type == 'sparsemoe':
            with torch.no_grad():
                mse_each_expert = self.mse_each_expert(pred_audio, gt['gt_data'])
                for i in range(mse_each_expert.shape[0]):
                    out[f'mse-expert-{i}_term'] = mse_each_expert[i]
        return out