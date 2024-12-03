# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# This code is the implementation of the Neural Experts model
# It is heavily based on our previous work DiGSimplementation with several significant modifications.
# It was partly based on SIREN and SAL implementation and architecture but with several significant modifications.
# for the original DiGS version see: https://github.com/Chumbyte/DiGS
# for the original SIREN version see: https://github.com/vsitzmann/siren
# for the original SAL version see: https://github.com/matanatz/SAL

import numpy as np
import torch
import torch.nn as nn
import models.managers as managers

from scipy.spatial.distance import cdist
from models.modules import (FullyConnectedNN, ParallelFullyConnectedNN, InputEncoder, tSoftMax, DummyModule)
from sklearn.cluster import KMeans

class Manager(nn.Module):
    def __init__(self, cfg, in_dim, module_name=''):
        super().__init__()
        self.n_experts = cfg['n_experts']
        self.point_dim = cfg['in_dim']
        self.temperature = cfg['manager_softmax_temperature']
        self.q_activation_dict = {'softmax': tSoftMax(self.temperature, -1, cfg['manager_softmax_temp_trainable']),
                                  'sigmoid': nn.Sigmoid(), 'none': DummyModule()}
        self.manager_type_dict = {'none': lambda: managers.DummyManager(self.n_experts),
                                  'standard': lambda: FullyConnectedNN(in_dim, self.n_experts,
                                                   num_hidden_layers=cfg['manager_n_hidden_layers'],
                                                   hidden_features=cfg['manager_hidden_dim'], outermost_linear=True,
                                                   nonlinearity=cfg['manager_nl'], init_type=cfg['manager_init'],
                                                   input_encoding=cfg['manager_input_encoding'],
                                                #    sphere_init_params=cfg['sphere_init_params'],
                                                #    init_r=cfg['manager_init_r'],
                                                   module_name=module_name + '.manager_net'),
                                  }

        self.manager_type = cfg['manager_type']
        self.manager_net = self.manager_type_dict.get(self.manager_type, lambda: None)()
        if self.manager_net is None:
            raise ValueError("Unsupported manager type")

        self.q_activation = self.q_activation_dict.get(cfg['manager_q_activation'], DummyModule())
        self.clamp_q = cfg['manager_clamp_q']

    def forward(self, points):
        raw_q = self.manager_net(points)

        q = self.q_activation(raw_q)
        q = torch.clamp(q, self.clamp_q)

        selected_expert_idx = torch.argmax(q, 1)
        return q, selected_expert_idx, raw_q.T.unsqueeze(0)


class INR_MoE(nn.Module):
    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all['MODEL']
        # self.encoder_type = cfg['encoder_type']
        self.init_type = cfg['decoder_init_type']
        self.n_experts = cfg['n_experts']
        self.manager_init = cfg['manager_init']
        self.share_encoder = cfg['shared_encoder']
        # self.manager_gt_input_sanitycheck = cfg['manager_gt_input_sanitycheck']

        self.decoder_input_encoding_module = InputEncoder(cfg, cfg['decoder_input_encoding'], cfg['decoder_hidden_dim'],
                                                          module_name='decoder_input_encoding_module')
        decoder_first_layer_dim = self.decoder_input_encoding_module.first_layer_dim

        self.decoder = ParallelFullyConnectedNN(self.n_experts, decoder_first_layer_dim, cfg['out_dim'],
                                        num_hidden_layers=cfg['decoder_n_hidden_layers'],
                                        hidden_features=cfg['decoder_hidden_dim'], outermost_linear=True,
                                        nonlinearity=cfg['decoder_nl'], init_type=self.init_type,
                                        input_encoding=cfg['decoder_input_encoding'],
                                        # sphere_init_params=cfg['sphere_init_params'], 
                                        # init_r=cfg['decoder_init_r'],
                                        freq=cfg['decoder_freqs'],
                                        module_name='decoder')  # SIREN decoder


        # in_dim = cfg['in_dim'] + latent_size
        # if self.manager_gt_input_sanitycheck:
        #     cfg['in_dim'] = cfg['in_dim'] + 3 # this is for the manager sanity check. remove later

        self.manager_conditioning = cfg['manager_conditioning']
        self.manager_conditioner = managers.ManagerConditioner(cfg['manager_conditioning'],decoder_first_layer_dim, self.decoder)
        if not self.share_encoder:
            self.manager_input_encoding_module = InputEncoder(cfg, cfg['manager_input_encoding'],
                                                              cfg['manager_hidden_dim'],
                                                              module_name='manager_input_encoding_module')
        manager_first_layer_dim = self.manager_input_encoding_module.first_layer_dim
        manager_first_layer_dim = manager_first_layer_dim + 1024 if cfg['manager_type'] == 'pointnet' else manager_first_layer_dim
        manager_first_layer_dim = manager_first_layer_dim + decoder_first_layer_dim if not self.manager_conditioning == 'none' else manager_first_layer_dim
        # cfg['planar_init_params']['idx'] = self.n_experts - 1 # TODO FIX THIS
        self.manager_net = Manager(cfg, manager_first_layer_dim)


    def forward(self, non_mnfld_pnts, mnfld_pnts=None, **kwargs):
        # non_mnfld_pnts: (1,n_nm,d), mnfld_pnts: (1, n_m, d)
        # d is input dim (2 or 3), k is num_experts
        encoded_non_mnfld_pnts_experts = self.decoder_input_encoding_module(non_mnfld_pnts, **kwargs)
        # Manifold points
        if mnfld_pnts is not None:
            encoded_mnfld_pnts_experts = self.decoder_input_encoding_module(mnfld_pnts, **kwargs)
            # mnfld_pnts_rep = mnfld_pnts.repeat_interleave(self.n_experts, dim=0).unsqueeze(0)
            manifold_pnts_pred = self.decoder(encoded_mnfld_pnts_experts)  # (1,k,n_m,1)
            manifold_pnts_pred = manifold_pnts_pred.squeeze(-1)  # (1,k,n_m)
            if not self.share_encoder:
                manager_input = self.manager_input_encoding_module(mnfld_pnts, **kwargs)

            manager_input = self.manager_conditioner(encoded_mnfld_pnts_experts, manager_input, **kwargs)

            mnfld_q, mnfld_selected_expert_idx, mnfld_raw_q, = self.manager_net(manager_input.view(-1, manager_input.shape[-1])) # (n_m,k), (n_m,)
            mnfld_q = mnfld_q.T.unsqueeze(0) # (1,k,n_m)

            selected_manifold_pnts_pred = torch.gather(manifold_pnts_pred, dim=-2,
                                                       index=mnfld_selected_expert_idx[None, None, :]).squeeze(-2) # (1,n_m)
        else:
            # mnfld_pnts_rep = None
            manifold_pnts_pred = None
            mnfld_q, mnfld_selected_expert_idx, selected_manifold_pnts_pred = None, None, None
            mnfld_raw_q = None

        # Off manifold points
        # non_mnfld_pnts_rep = non_mnfld_pnts.repeat_interleave(self.n_experts, dim=0).unsqueeze(0)
        nonmanifold_pnts_pred = self.decoder(encoded_non_mnfld_pnts_experts) # (1,k,n_nm,d)
        # nonmanifold_pnts_pred = nonmanifold_pnts_pred.squeeze(-1) # (1,k,n_nm)

        # if self.manager_gt_input_sanitycheck:
        #     non_mnfld_pnts = torch.cat([non_mnfld_pnts, kwargs['img']], dim=-1)

        if not self.share_encoder:
            encoded_non_mnfld_pnts_manager = self.manager_input_encoding_module(non_mnfld_pnts,**kwargs)


        nonmnld_manager_input = encoded_non_mnfld_pnts_manager # (1,n_nm,d)

        nonmnld_manager_input = self.manager_conditioner(encoded_non_mnfld_pnts_experts, nonmnld_manager_input, **kwargs) # (1,n_nm,d)

        nonmnld_manager_input = nonmnld_manager_input.reshape(-1, nonmnld_manager_input.shape[-1])
        mout_nonmnfld_q, nonmnfld_selected_expert_idx, nonmnfld_raw_q, = self.manager_net(nonmnld_manager_input) # (n_nm, k), (n_nm,)
        nonmnfld_q = mout_nonmnfld_q.T.unsqueeze(0) # (1,k,n_nm)


        selected_nonmanifold_pnts_pred = torch.gather(nonmanifold_pnts_pred, dim=-3,
                                                      index=nonmnfld_selected_expert_idx[None, None, :, None].repeat([1, 1, 1, nonmanifold_pnts_pred.shape[-1]])).squeeze(-2) # (1, n_nm)

        return {"manifold_pnts_pred": manifold_pnts_pred,                           # (1,k,n_m)
                "nonmanifold_pnts_pred": nonmanifold_pnts_pred,                     # (1,k,n_nm)
                "mnfld_q": mnfld_q,                                                 # (1,k,n_m)
                "nonmnfld_q": nonmnfld_q,                                           # (1,k,n_nm)
                "selected_manifold_pnts_pred": selected_manifold_pnts_pred,         # (1,n_m)
                "selected_nonmanifold_pnts_pred": selected_nonmanifold_pnts_pred,   # (1,n_nm)
                "mnfld_selected_expert_idx": mnfld_selected_expert_idx,             # (n_m,)
                "nonmnfld_selected_expert_idx": nonmnfld_selected_expert_idx,       # (n_nm,)
                "mnfld_raw_q": mnfld_raw_q,
                "nonmnfld_raw_q": nonmnfld_raw_q,
                "nonmnld_manager_input": nonmnld_manager_input,
                "mout_nonmnfld_q": mout_nonmnfld_q,
                }




