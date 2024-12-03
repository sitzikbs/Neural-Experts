# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# This file contains the loss dunctions for the RGB image reconstruction task

import torch.nn as nn
import utils.utils as utils
import models.utils as model_utils
import torch
import torch.nn.functional as F


###################################### image reconstruction Loss #############################################
class DummyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute_loss(self, *args, **kwargs):
        return torch.tensor(0)
    def forward(self, *args, **kwargs):
        return torch.tensor(0)


class RGBReconLossMoE(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    def compute_loss(self, pred_vals, gt_vals, q, *args, **kwargs):

        if self.metric == 'hinton':
            delta = ((pred_vals - gt_vals.unsqueeze(-1)) ** 2).sum(-2)
            rgb_recon_loss = -torch.log((q*torch.exp(-0.5*delta)).sum(-1)).mean()
        else:
            rgb_recon_loss = (((pred_vals - gt_vals.unsqueeze(-1)) ** 2)*q.unsqueeze(-2)).mean()

        return rgb_recon_loss


class RGBReconLossSingle(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_loss(self, pred_vals, gt_vals, *args, **kwargs):
        rgb_recon_loss = ((pred_vals - gt_vals) ** 2).mean()
        return rgb_recon_loss



class RGBReconLoss(nn.Module):
    def __init__(self, loss_type, metric):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': RGBReconLossSingle(),
                     'moe': RGBReconLossMoE(metric),
                     'sparsemoe': RGBReconLossMoE(metric)}
        self.loss = loss_dict[loss_type]

    def forward(self, pred_vals, gt_vals, q, *args, **kwargs):
        return self.loss.compute_loss(pred_vals, gt_vals, q)

class PSNR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, I, mse, *args, **kwargs):
        psnr = 10 * torch.log10(I.max().square() / mse)
        return psnr

class PSNREachExpert(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_vals, gt_vals, *args, **kwargs):
        mse = ((pred_vals - gt_vals.unsqueeze(-1)) ** 2).mean(1).mean(1).squeeze()
        psnr_each = 10 * torch.log10(gt_vals.max().square() / mse)
        return psnr_each


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
        if self.seg_type == 'ce':
            segmentation_loss = F.cross_entropy(q, gt_segment.long(), label_smoothing=0.01)
        elif self.seg_type == 'binary_ce':
            segmentation_loss = F.binary_cross_entropy_with_logits(q, torch.ones_like(q.squeeze()) / self.n_experts)
        elif self.seg_type == 'both':
            segmentation_loss = (F.cross_entropy(q, gt_segment.long(), label_smoothing=0.01) + 0.1*
                                 F.binary_cross_entropy_with_logits(q, torch.ones_like(q.squeeze()) / self.n_experts))
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

###################################### clustering Loss #############################################

class ClusteringLossMoE(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_loss(self, coords, q, eps=1e-6, *args, **kwargs):
        in_dim = coords.shape[-1]
        q_sum = q.unsqueeze(-1).sum(1, keepdim=True)
        cluster_center = (q.unsqueeze(-1).expand(-1, -1, -1, in_dim) * coords.unsqueeze(-2)).sum(1, keepdim=True) / q_sum
        distance_to_center = (coords.unsqueeze(2) - cluster_center.expand(-1, coords.shape[1], -1, -1)).norm(2, dim=-1).square()
        clustering_loss = (distance_to_center * q / q_sum.squeeze(-1)).sum(1).mean() # should this be devided by q_sum?
        return clustering_loss


class ClusteringLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': DummyLoss(),
                     'moe': ClusteringLossMoE(),
                     'sparsemoe': ClusteringLossMoE()}
        self.loss = loss_dict[loss_type]

    def forward(self, coords, q, *args, **kwargs):
        return self.loss.compute_loss(coords, q)


###################################### Eikonal Loss #############################################
class EikonalLossSingle(nn.Module):
    def __init__(self, metric_type='abs'):
        super().__init__()
        self.metric_type = metric_type

    def compute_loss(self, output, input, *args, **kwargs):
        # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
        # shape is (bs, num_points, dim=3) for both grads
        all_grads = utils.gradient(input, output)

        if self.metric_type == 'abs':
            eikonal_term = ((all_grads.norm(2, dim=-1) - 1).abs()).mean()
        else:
            eikonal_term = ((all_grads.norm(2, dim=-1) - 1).square()).mean()

        return eikonal_term

class EikonalLoss(nn.Module):
    def __init__(self, loss_type, metric_type='abs'):
        super().__init__()
        self.metric_type = metric_type
        loss_dict = {'none': DummyLoss(),
                     'single': DummyLoss(),
                     'moe':  EikonalLossSingle(self.metric_type),
                     'sparsemoe': EikonalLossSingle(self.metric_type)}

        self.loss = loss_dict[loss_type]
    def forward(self, output, input,  *args, **kwargs):
        return self.loss.compute_loss(output, input)


###################################### Eikonal Loss #############################################
class LipschitzLossSingle(nn.Module):
    def __init__(self, metric_type='abs'):
        super().__init__()
        self.metric_type = metric_type

    def compute_loss(self, model, *args, **kwargs):
        # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
        # shape is (bs, num_points, dim=3) for both grads
        lipc = 1.0
        for name, param in model.named_parameters():
            if 'c_list' in name:
                lipc = lipc * F.softplus(param.squeeze())

        return lipc

class LipschitzLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': DummyLoss(),
                     'moe':  LipschitzLossSingle(),
                     'sparsemoe': LipschitzLossSingle()}

        self.loss = loss_dict[loss_type]
    def forward(self, model,  *args, **kwargs):
        return self.loss.compute_loss(model)

###################################### q entropy Loss #############################################
class QEntropyLossMoE(nn.Module):
    def __init__(self, metric='entropy', n_experts=1):
        super().__init__()
        self.metric = metric
        assert self.metric == 'entropy' or self.metric == 'kl'
        if self.metric == 'kl':
            self.kl_div = nn.KLDivLoss(reduction='batchmean')
            self.n_experts = n_experts


    def compute_loss(self, q, *args, **kwargs):
        if self.metric == 'entropy':
            # minimize the entropy over all sampled points to push the q distribution to be bimodal
            q = F.normalize(q, p=2, dim=1) # normalize q point wise
            q_entropy_loss = (q * torch.log(q + 1e-6)).sum(-1).mean()
        elif self.metric == 'kl':
            q_hat = F.normalize(q, p=2, dim=1)  # normalize q point wise
            q_entropy_loss1 = -(q_hat * torch.log(q_hat + 1e-6)).sum(-1).mean() # speard it out
            q_entropy_loss2 = self.kl_div(q.squeeze().log(), torch.ones_like(q.squeeze()) / self.n_experts) #make it uniform
            q_entropy_loss = 10*q_entropy_loss1 + 0.01*q_entropy_loss2

        return q_entropy_loss

class QEntropyLoss(nn.Module):
    def __init__(self, loss_type, metric='entropy', n_experts=1):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': DummyLoss(),
                     'moe':  QEntropyLossMoE(metric, n_experts),
                     'sparsemoe': QEntropyLossMoE(metric, n_experts)}

        self.loss = loss_dict[loss_type]
    def forward(self, q,  *args, **kwargs):
        return self.loss.compute_loss(q)


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

###################################### sum to one Loss #############################################
class SumToOneMoE(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_loss(self, q, *args, **kwargs):
        loss = (q.sum(-1) - 1).abs().mean()
        return loss

class SumToOneLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': DummyLoss(),
                     'moe':  SumToOneMoE(),
                     'sparsemoe':  SumToOneMoE()}

        self.loss = loss_dict[loss_type]
    def forward(self, q,  *args, **kwargs):
        return self.loss.compute_loss(q)



###################################### RGBImage applied to all experts without q Loss #############################################
class RGBReconAllLossMoE(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_loss(self, pred_vals, gt_vals, *args, **kwargs):
        rgb_recon_loss = (((pred_vals - gt_vals.unsqueeze(-1)) ** 2)).mean()
        return rgb_recon_loss

class RGBReconAllLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': RGBReconLossSingle(),
                     'moe': RGBReconAllLossMoE(),
                     'sparsemoe': RGBReconAllLossMoE()}
        self.loss = loss_dict[loss_type]

    def forward(self, pred_vals, gt_vals, *args, **kwargs):
        return self.loss.compute_loss(pred_vals, gt_vals)


###################################### Expert variance Loss #############################################
class ExpertVarLossMoE(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_loss(self, pred_vals, *args, **kwargs):

        intravar = 0.001/torch.clamp(pred_vals.var(dim=-1).norm(dim=-1).mean(), 0.0001) # push the color variance within the expert to be small
        intervar = pred_vals.var(dim=1).norm(dim=-2).mean() # push the color variance between the experts to be large

        expert_var_loss = intervar + intravar
        return expert_var_loss

class ExpertVarLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': DummyLoss(),
                     'moe': ExpertVarLossMoE(),
                     'sparsemoe': ExpertVarLossMoE()}
        self.loss = loss_dict[loss_type]

    def forward(self, pred_vals, *args, **kwargs):
        return self.loss.compute_loss(pred_vals)


###################################### Manager Auxilary Loss #############################################
class ManagerAuxLossMoE(nn.Module):
    def __init__(self):
        super().__init__()


    def compute_loss(self, aux_vals, gt_vals, *args, **kwargs):
        aux_loss = ((aux_vals - gt_vals) ** 2).mean()
        return aux_loss

class AuxLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {'none': DummyLoss(),
                     'single': DummyLoss(),
                     'moe': ManagerAuxLossMoE(),
                     'sparsemoe': ManagerAuxLossMoE()}
        self.loss = loss_dict[loss_type]

    def forward(self, aux_vals, gt_vals, *args, **kwargs):
        return self.loss.compute_loss(aux_vals, gt_vals)


LOSS_LIST = ['rgbrecon', 'rgbreconall', 'balance', 'segmentation', 'clustering', 'eikonal', 'lipschitz', 'qentropy', 'load', 'sumtoone', 'expertvar', 'aux']
class RGBImageLoss(nn.Module):
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

        self.rgbrecon_loss = RGBReconLoss(self.required_loss_dict['rgbrecon'], cfg['rgbrecon_metric'])
        self.rgbreconall_loss = RGBReconAllLoss(self.required_loss_dict['rgbreconall'])
        self.balance_loss = BalancingLoss(self.required_loss_dict['balance'], n_experts, self.sample_bias_correction)
        self.segmentation_loss = SegmentationLoss(self.required_loss_dict['segmentation'], n_experts, cfg['segmentation_type'])
        self.clustering_loss = ClusteringLoss(self.required_loss_dict['clustering'])
        self.eikonal_loss = EikonalLoss(self.required_loss_dict['eikonal'])
        self.lipschitz_loss = LipschitzLoss(self.required_loss_dict['lipschitz'])
        self.qentropy_loss = QEntropyLoss(self.required_loss_dict['qentropy'], cfg['entropy_metric'], n_experts)
        self.load_loss = LoadLoss(self.required_loss_dict['load'])
        self.q_entropy_metric_str = cfg['entropy_metric']
        self.sumtoone_loss = SumToOneLoss(self.required_loss_dict['sumtoone'])
        self.expertvar_loss = ExpertVarLoss(self.required_loss_dict['expertvar'])
        self.aux_loss = AuxLoss(self.required_loss_dict['aux'])



        self.recon_mse = RGBReconLossSingle()
        self.psnr = PSNR()
        self.psnr_each_expert = PSNREachExpert()


    def forward(self, output_pred, coords, gt={'img': None, 'segments': None, 'aux': None}, model=None):
        q = None
        if self.model_type == 'moe' or self.model_type == 'sparsemoe' or self.model_type == 'sparseidfmoe':
            q = output_pred['nonmnfld_q'].permute(0, 2, 1)
            raw_q = output_pred['nonmnfld_raw_q'].permute(0, 2, 1).squeeze()
            final_img = output_pred['selected_nonmanifold_pnts_pred']
            gt_segments = gt['segment'].squeeze()
            if self.model_type == 'sparsemoe':
                nonmnfld_selected_expert_idx = output_pred['nonmnfld_selected_expert_idx']
                q = torch.gather(q, dim=-1, index=nonmnfld_selected_expert_idx[None, None, :])# (1, n_nm)
        else:
            final_img = output_pred['nonmanifold_pnts_pred'].permute(0, 2, 1)
            raw_q, gt_segments = None, None

        if output_pred['nonmanifold_pnts_pred'].dim() == 4:
            pred_img = output_pred['nonmanifold_pnts_pred'].permute(0, 2, 3, 1)
        else:
            pred_img = output_pred['nonmanifold_pnts_pred'].permute(0, 2, 1)

        if self.sample_bias_correction: # scale q by the number of pixel it "sees" so computing the mean is co
            q = q * q.shape[1] / torch.clamp(q.sum(-2, keepdim=True), 0.00001)

        rgbrecon_term = self.rgbrecon_loss(pred_img, gt['img'], q)
        rgbreconall_term = self.rgbreconall_loss(pred_img, gt['img'])
        balance_term = self.balance_loss(q)
        segmentation_term = self.segmentation_loss(raw_q, gt_segments)
        clustering_term = self.clustering_loss(coords, q)
        eikonal_term = self.eikonal_loss(q, coords)
        lipschitz_term = self.lipschitz_loss(model)
        qentropy_term = self.qentropy_loss(q)
        load_term = self.load_loss(output_pred.get('importance', None), output_pred.get('load', None), q)
        sumtoone_term = self.sumtoone_loss(q)
        expert_var_term = self.expertvar_loss(pred_img)
        aux_term = self.aux_loss(output_pred.get('nonmnfld_aux', None), gt['aux'])

        loss = (rgbrecon_term * self.weight_dict['rgbrecon'] + balance_term * self.weight_dict['balance'] +
                segmentation_term * self.weight_dict['segmentation'] + clustering_term * self.weight_dict['clustering']+
                eikonal_term * self.weight_dict['eikonal'] + lipschitz_term * self.weight_dict['lipschitz'] +
                qentropy_term * self.weight_dict['qentropy'] + load_term * self.weight_dict['load'] +
                sumtoone_term * self.weight_dict['sumtoone'] + rgbreconall_term * self.weight_dict['rgbreconall'] +
                expert_var_term * self.weight_dict['expertvar'] + aux_term * self.weight_dict['aux'])

        # compute error measures
        with torch.no_grad():
            recon_error = self.recon_mse.compute_loss(final_img.squeeze(), gt['img'].squeeze())
            psnr = self.psnr(gt['img'], recon_error)


        out = {'loss': loss, 'rgbrecon_term': rgbrecon_term, 'balance_term': balance_term,
                'reconerror_term': recon_error, 'psnr_term': psnr, 'segmentation_term': segmentation_term,
                'clustering_term': clustering_term, 'eikonal_term': eikonal_term, 'lipschitz_term': lipschitz_term,
                'qentropy_term': qentropy_term, 'load_term': load_term, 'sumtoone_term': sumtoone_term,
                'rgbreconall_term': rgbreconall_term, 'expertvar_term': expert_var_term, 'aux_term': aux_term}

        if not self.model_type == 'single' and not self.model_type == 'sparsemoe':
            with torch.no_grad():
                psnr_each_expert = self.psnr_each_expert(pred_img, gt['img'])
                for i in range(psnr_each_expert.shape[0]):
                    out[f'psnr-expert-{i}_term'] = psnr_each_expert[i]
        return out


