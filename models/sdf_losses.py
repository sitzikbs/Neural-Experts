import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils
import numpy as np
import matplotlib.pyplot as plt

import utils.utils as utils
import models.utils as model_utils

class DummyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute_loss(self, *args, **kwargs):
        return torch.tensor(0)
    def forward(self, *args, **kwargs):
        return torch.tensor(0)

###################################### Zero Level Set Loss #############################################
class ZLSLossSingle(nn.Module):
    def __init__(self):
        super().__init__()
    def compute_loss(self, manifold_pred, *args, **kwargs):
        return torch.abs(manifold_pred).mean()

class ZLSLossMoE(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_loss(self, manifold_pred, q, *args, **kwargs):
        return (torch.abs(manifold_pred)*q).mean()


class ZLSLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': ZLSLossSingle(),
                     'moe': ZLSLossMoE()}
        self.loss = loss_dict[loss_type]

    def forward(self, manifold_pred, q, *args, **kwargs):
        return self.loss.compute_loss(manifold_pred, q)

###################################### SDF Loss #############################################
class SDFLossSingle(nn.Module):
    def __init__(self, sdf_clamp=0.1):
        super().__init__()
        self.sdf_clamp = sdf_clamp
    def compute_loss(self, sdf_pred, sdf_gt, q, *args, **kwargs):
        clamped_sdf_pred = torch.clamp(sdf_pred, -self.sdf_clamp, self.sdf_clamp)
        clamped_sdf_gt = torch.clamp(sdf_gt, -self.sdf_clamp, self.sdf_clamp)
        sdf_term = torch.abs(clamped_sdf_pred - clamped_sdf_gt) # (1,n)
        return sdf_term.mean()


class SDFLossMoE(nn.Module):
    def __init__(self, sdf_clamp=0.1):
        super().__init__()
        self.sdf_clamp = sdf_clamp

    def compute_loss(self, sdf_pred, sdf_gt, q, *args, **kwargs):
        if sdf_gt is not None:
            clamped_sdf_pred = torch.clamp(sdf_pred, -self.sdf_clamp, self.sdf_clamp)
            clamped_sdf_gt = torch.clamp(sdf_gt, -self.sdf_clamp, self.sdf_clamp)
            sdf_term = (torch.abs(clamped_sdf_pred - clamped_sdf_gt.unsqueeze(1))*q).mean()
        else:
            sdf_term = 0

        return sdf_term


class SDFLoss(nn.Module):
    def __init__(self, loss_type, sdf_clamp=0.1):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': SDFLossSingle(sdf_clamp),
                     'moe': SDFLossMoE(sdf_clamp)}
        self.loss = loss_dict[loss_type]

    def forward(self, sdf_pred, sdf_gt, q, *args, **kwargs):
        return self.loss.compute_loss(sdf_pred, sdf_gt, q)

###################################### Segmentation Loss #############################################

class SegmentationLossMoE(nn.Module):
    def __init__(self, n_experts, seg_type='ce'):
        super().__init__()
        self.n_experts = n_experts
        self.seg_type = seg_type

    def compute_loss(self, q, gt_segment, *args, **kwargs):
        if self.seg_type == 'ce':
            segmentation_loss = F.cross_entropy(q, gt_segment.long(), label_smoothing=0.01)
        elif self.seg_type == 'binary_ce':
            segmentation_loss = F.binary_cross_entropy_with_logits(q, torch.ones_like(q.squeeze()) / self.n_experts)
        elif self.seg_type == 'both':
            segmentation_loss = (F.cross_entropy(q, gt_segment.long(), label_smoothing=0.01) + 0.1*
                                 F.binary_cross_entropy_with_logits(q, torch.ones_like(q.squeeze()) / self.n_experts))
            # segmentation_loss = (F.cross_entropy(q, gt_segment.long()) + 0.1 *
            #                      F.binary_cross_entropy_with_logits(q, torch.ones_like(q.squeeze()) / self.n_experts))
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

LOSS_LIST = ['zls', 'sdf', 'segmentation']  # do not use underscore! it is used as a separator in the loss string

class SDFShapeLoss(nn.Module):
    def __init__(self, cfg, model_name, in_dim=2, model=None, n_experts=1):
        super().__init__()
        moe_indicator = '_moe' in model_name
        self.model_type = 'moe' if moe_indicator else 'single'
        self.model = model
        self.gradient_comp = utils.experts_gradient if moe_indicator else utils.gradient
        self.weight_dict = {}
        self.cfg = cfg

        required_loss_list, weights = model_utils.parse_loss_string(cfg['loss_type'])
        self.required_loss_dict, self.weight_dict = model_utils.build_loss_dictionary(required_loss_list, weights, self.model_type, full_loss_list=LOSS_LIST)

        self.zls_loss = ZLSLoss(self.required_loss_dict['zls'])
        self.sdf_loss = SDFLoss(self.required_loss_dict['sdf'], cfg['sdf_clamp'])
        self.segmentation_loss = SegmentationLoss(self.required_loss_dict['segmentation'], n_experts, cfg['segmentation_type'])

    def update_weight(self, current_iteration, n_iterations, loss_name='div', decay_type='linear', params=None):
        # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.75, 1] of the training process, the weight should
        #   be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
        #   Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1]))

        curr = current_iteration / n_iterations
        we, e = min([tup for tup in decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        # Divergence term anealing functions
        if decay_type == 'linear':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weight_dict[loss_name] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weight_dict[loss_name] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
            else:
                self.weight_dict[loss_name]  = we
        elif decay_type == 'quintic':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weight_dict[loss_name] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weight_dict[loss_name]  = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
            else:
                self.weight_dict[loss_name]  = we
        elif decay_type == 'step':  # change weight at s
            if current_iteration < s * n_iterations:
                self.weight_dict[loss_name]  = w0
            else:
                self.weight_dict[loss_name]  = we
        elif decay_type == 'none':
            pass
        else:
            raise Warning("unsupported div decay value")

    # @profile
    def forward(self, output_pred, data=None, dataset=None):
        # 'mnfld_points': mnfld_points, # (n_points, d)
        # "nonmnfld_points": nonmnfld_points,  # (2*n_points, d)
        # "nonmnfld_sdf": nonmnfld_sdf,  # (2*n_points, )
        # 'nonmnfld_segments_gt': nonmnfld_segments,  # (2*n_points, )

        if self.model_type == 'moe' and 'nonmnfld_segments_gt' in data:
            # When segmentation_mode is True

            raw_q = output_pred['nonmnfld_raw_q'] # (1,k,n)
            segments = data['nonmnfld_segments_gt'].squeeze()
            segmentation_term = self.segmentation_loss(raw_q.permute(0, 2, 1).squeeze(), segments)
            
            loss = self.weight_dict['segmentation'] * segmentation_term
            if torch.isnan(loss):
                import pdb; pdb.set_trace()

            loss_dict = {"loss": loss, 
                    'segmentation_term': segmentation_term,
                    }
            
            return loss_dict
        
        mnfld_points = data['mnfld_points']
        nonmnfld_points = data['nonmnfld_points']
        nonmnfld_sdf_gt = data['nonmnfld_sdf']
        nonmnfld_q = None
        mnfld_q = None

        if self.model_type == 'moe':
            mnfld_q = output_pred["mnfld_q"]
            nonmnfld_q = output_pred["nonmnfld_q"]
            raw_q = output_pred['nonmnfld_raw_q'] # (1,k,n)
            manifold_pred = output_pred["manifold_pnts_pred"] # (1, k, n_m)
            non_manifold_pred = output_pred['nonmanifold_pnts_pred'].squeeze(-1) # (1, k, n_nm)
            # final_manifold_pred = output_pred["selected_manifold_pnts_pred"] # (1, n_nm)
            # final_non_manifold_pred = output_pred["selected_nonmanifold_pnts_pred"].squeeze(1).squeeze(-1) # (1, n_nm)
        else:
            manifold_pred = output_pred["manifold_pnts_pred"].squeeze(-1) # (1, n_m)
            non_manifold_pred = output_pred['nonmanifold_pnts_pred'].squeeze(1) # (1,n_nm)
            # final_manifold_pred = manifold_pred
            # final_non_manifold_pred = non_manifold_pred

        zls_term = self.zls_loss(manifold_pred, mnfld_q)
        sdf_term = self.sdf_loss(non_manifold_pred, nonmnfld_sdf_gt, nonmnfld_q)
        
        loss = self.weight_dict['zls'] * zls_term + self.weight_dict['sdf'] * sdf_term
        
        loss_dict = {"loss": loss, 
                     'zls_term': zls_term, 
                     'sdf_term': sdf_term, 
                     }

        epoch = output_pred['epoch']
        if self.model_type == 'moe' and epoch % 500 == 0:
            # t0 = time.time()
            logdir = output_pred["logdir"]
            out_dir = os.path.join(logdir, 'out')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'q_dist_{epoch:05d}.png')
            q_vals = raw_q.permute(0, 2, 1).squeeze().detach().cpu()
            q_normalised = F.softmax(q_vals, dim=-1)

            _, n_experts = q_vals.shape
            n_cols = 3
            n_rows = np.ceil(n_experts/n_cols).astype(int)
            if epoch % 500 == 0:
                fig = plt.figure(1, (20, 20))
                axs = []
                for expert_id in range(n_experts):
                    q_expert = q_normalised[expert_id]
                    ax = fig.add_subplot(n_rows, n_cols, expert_id + 1)
                    axs.append(ax)
                    ax.set_title('Expert {} q distribution'.format(expert_id))
                    frq, edges = np.histogram(q_expert.flatten(), bins=100)
                    ax.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
                    # Set axis limits and labels
                    ax.set_xlim(0, 1)
                    # ax.set_ylim(0, 5000)
                plt.savefig(out_path)
        
        return loss_dict

