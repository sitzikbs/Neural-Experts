# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# This file contains the code for different types of layer initializations

import torch
import torch.nn as nn
import numpy as np


################################# SIREN's initialization ###################################
class SirenInit(nn.Module):
    def __init__(self):
        super().__init__()

    def sine_init(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See SIREN paper supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


    def first_layer_sine_init(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-1 / num_input, 1 / num_input)

    def forward(self, net):
        net.apply(self.sine_init)
        net[0].apply(self.first_layer_sine_init)
        return net


################################# sine geometric initialization ###################################
class SirenGeomInit(nn.Module):
    def __init__(self, flip=False, r=0.5, centroid=None):
        super().__init__()
        self.centroids = centroid
        self.r = r
        self.flip = 1
        if flip:
            self.flip = -1
    def geom_sine_init(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_output = m.weight.size(0)
                m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
                m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
                m.weight.data /= 30
                m.bias.data /= 30

    def first_layer_geom_sine_init(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_output = m.weight.size(0)
                m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
                m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
                if self.centroids is not None:
                    bias_term = -m.weight.data.matmul(self.centroids)
                    m.bias.data = bias_term + m.bias.data
                m.weight.data /= 30
                m.bias.data /= 30


    def second_last_layer_geom_sine_init(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_output = m.weight.size(0)
                assert m.weight.shape == (num_output, num_output)
                m.weight.data = 0.5 * np.pi * torch.eye(num_output) + 0.001 * torch.randn(num_output, num_output)
                m.bias.data = 0.5 * np.pi * torch.ones(num_output, ) + 0.001 * torch.randn(num_output)
                m.weight.data /= 30
                m.bias.data /= 30

    def last_layer_geom_sine_init(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                n_experts = m.weight.size(0)
                assert m.weight.shape == (n_experts, num_input)
                assert m.bias.shape == (n_experts,)
                # m.weight.data = -1 * torch.ones(1, num_input) + 0.001 * torch.randn(num_input)
                m.weight.data = -1 * torch.ones(n_experts, num_input) + 0.00001 * torch.randn(num_input)
                m.bias.data = torch.zeros(n_experts) + num_input
                m.weight.data = self.flip*m.weight.data #flip to have the positive be inside the spheere
                m.bias.data = self.flip*m.bias.data + self.r

    def forward(self, net):
        net.apply(self.geom_sine_init)
        net[0].apply(self.first_layer_geom_sine_init)
        net[-2].apply(self.second_last_layer_geom_sine_init)
        net[-1].apply(self.last_layer_geom_sine_init)
        return net

################################# multi frequency geometric initialization ###################################
class MFGIInit(nn.Module):
    def __init__(self):
        super(SirenGeomInit).__init__()
        self.periods = [1, 30] # Number of periods of sine the values of each section of the output vector should hit
        # periods = [1, 60] # Number of periods of sine the values of each section of the output vector should hit
        self.portion_per_period = np.array([0.25, 0.75]) # Portion of values per section/period

    def first_layer_mfgi_init(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                num_output = m.weight.size(0)
                num_per_period = (self.portion_per_period * num_output).astype(int) # Number of values per section/period
                assert len(self.periods) == len(num_per_period)
                assert sum(num_per_period) == num_output
                weights = []
                for i in range(0, len(self.periods)):
                    period = self.periods[i]
                    num = num_per_period[i]
                    scale = 30/period
                    weights.append(torch.zeros(num,num_input).uniform_(-np.sqrt(3 / num_input) / scale, np.sqrt(3 / num_input) / scale))
                W0_new = torch.cat(weights, axis=0)
                m.weight.data = W0_new

    def second_layer_mfgi_init(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                assert m.weight.shape == (num_input, num_input)
                num_per_period = (self.portion_per_period * num_input).astype(int) # Number of values per section/period
                k = num_per_period[0] # the portion that only hits the first period
                # W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input), np.sqrt(3 / num_input) / 30) * 0.00001
                W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input), np.sqrt(3 / num_input) / 30) * 0.0005
                W1_new_1 = torch.zeros(k, k).uniform_(-np.sqrt(3 / num_input) / 30, np.sqrt(3 / num_input) / 30)
                W1_new[:k, :k] = W1_new_1
                m.weight.data = W1_new

    def forward(self, net):
        net.apply(self.geom_sine_init)
        net[0].apply(self.first_layer_mfgi_init)
        net[1].apply(self.second_layer_mfgi_init)
        net[-2].apply(self.second_last_layer_geom_sine_init)
        net[-1].apply(self.last_layer_geom_sine_init)
        return net


################################# geometric initialization used in SAL and IGR ###################################
class GeomReluInit(nn.Module):
    def __init__(self, flip=False, r=0.5, centroid=None):
        super().__init__()
        self.centroid = centroid
        self.r = r
        self.flip = 1
        if flip:
            self.flip = -1
    def geom_relu_init(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                out_dims = m.out_features

                m.weight.normal_(mean=0.0, std=np.sqrt(2) / np.sqrt(out_dims))
                m.bias.data = torch.zeros_like(m.bias.data)

    def geom_relu_last_layers_init(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.normal_(mean=np.sqrt(np.pi) / np.sqrt(num_input), std=0.00001)
                m.bias.data = torch.zeros_like(m.bias.data)
                m.weight.data = self.flip*m.weight.data #flip to have the positive be inside the spheere
                m.bias.data = m.bias.data - self.flip*self.r

    def geom_relu_first_layer_init(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                out_dims = m.out_features
                m.weight.normal_(mean=0.0, std=np.sqrt(2) / np.sqrt(out_dims))
                m.bias.data = torch.zeros_like(m.bias.data)
                if self.centroid is not None:
                    bias_term = -m.weight.data.matmul(self.centroid)
                    m.bias.data = bias_term + m.bias.data
    def forward(self, net):
        net.apply(self.geom_relu_init)
        net[0].apply(self.geom_relu_first_layer_init)
        net[-1].apply(self.geom_relu_last_layers_init)
        return net

class GeomReluConstInit(nn.Module):
    def __init__(self):
        super().__init__()
    def geom_relu_init_const(self, m):
        torch.manual_seed(0)
        with torch.no_grad():
            if hasattr(m, 'weight'):
                out_dims = m.out_features

                m.weight.normal_(mean=0.0, std=np.sqrt(2) / np.sqrt(out_dims))
                m.bias.data = torch.zeros_like(m.bias.data)

    def geom_relu_last_layers_init_const(self, m):
        torch.manual_seed(0)
        radius_init = 1
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.normal_(mean=np.sqrt(np.pi) / np.sqrt(num_input), std=0.00001)
                m.bias.data = torch.Tensor([-radius_init])

    def forward(self, net):
        net.apply(self.geom_relu_init_const)
        net[-1].apply(self.geom_relu_last_layers_init_const)
        return net

class NormalInit(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights_normal(self, m):
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, net):
        net.apply(self.init_weights_normal)
        return net


class KeimingInit(nn.Module):
    def __init__(self):
        super().__init__()
    def init_weights_normal(self, m):
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, net):
        net.apply(self.init_weights_normal)
        return net

################################# Planar initializationfor relu / softplus activation ###################################

class PlanarInit(nn.Module):
    def __init__(self, centroids, normals, dim, idx):
        super().__init__()
        self.centroids = centroids
        self.normals = normals.to(torch.float32) if normals is not None else None
        self.dim = dim
        self.idx = idx
        self.down_scale = 1000

    def init_weights_planar(self, m):
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
            m.weight.data = m.weight.data / self.down_scale
            m.bias.data = m.bias.data / self.down_scale
            m.weight.data[0, :] = m.weight.data[0, :] + 1

    def init_weights_planar_first_layer(self, m):
        if hasattr(m, 'weight'):
            outdim = m.weight.shape[0]
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
            m.weight.data = m.weight.data / self.down_scale
            m.bias.data = m.bias.data / self.down_scale
            m.weight.data = (1./outdim)*self.normals[self.idx] + m.weight.data
            bias_term = -(1./outdim)*torch.sum(self.centroids[self.idx] * self.normals[self.idx], dtype=torch.float32)
            m.bias.data = bias_term + m.bias.data

    def forward(self, net):
        net[1:].apply(self.init_weights_planar)
        net[0].apply(self.init_weights_planar_first_layer)
        return net


################################# Pretrained initialization ###################################
class PTInit(nn.Module):
    def __init__(self, centroids, normals, dim, idx, pt_path):
        super().__init__()
        self.centroids = centroids
        self.normals = normals.to(torch.float32) if normals is not None else None
        self.dim = dim
        self.idx = idx
        self.pt_path = pt_path

    def transform_weights(self, m):
        #TODO extend to 3D case
        if hasattr(m, 'weight'):
            outdim = m.weight.shape[0]
            rot_mat = torch.tensor([[self.normals[self.idx][1], self.normals[self.idx][0]],
                                    [-self.normals[self.idx][0], self.normals[self.idx][1]]], dtype=torch.float32,
                                   device=self.normals.device)
            m.weight.data = rot_mat.matmul(m.weight.data.T).T
            # bias_term = -torch.sum(self.centroids[self.idx] * self.normals[self.idx], dtype=torch.float32)
            bias_term = -m.weight.data.matmul(self.centroids[self.idx])
            m.bias.data = bias_term + m.bias.data

    def forward(self, net):
        #load pt model and extract weights (architecture with different names)
        pre_trained_model = torch.load(self.pt_path)
        pt_arch_weights = list(pre_trained_model.items())
        my_model_kvpair = net.state_dict()
        count = 0
        for key, value in my_model_kvpair.items():
            layer_name, weights = pt_arch_weights[count]
            my_model_kvpair[key] = weights
            count += 1
        net.load_state_dict(my_model_kvpair)
        net[0].apply(self.transform_weights)
        return net

class BasicPTInit(nn.Module):
    def __init__(self, pt_path):
        super().__init__()
        self.pt_path = pt_path

    def forward(self, module):
        # load pt model and extract weights (architecture with different names)
        pre_trained_model = torch.load(self.pt_path)
        pt_arch_weights = list(pre_trained_model.items())
        my_model_kvpair = module.state_dict()
        for key, value in my_model_kvpair.items():
            for layer_name, weights in pt_arch_weights:
                my_module_name = module.module_name  + '.' +  key
                if my_module_name == layer_name:
                    my_model_kvpair[key] = weights
        module.load_state_dict(my_model_kvpair)
        return module

def pytorchWeightInit(decoder):
    # default weight initialization in PyTorch for nn.Linear
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        nn.init.kaiming_uniform_(decoder.weights_list[i], a=np.sqrt(5))
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(decoder.biases_list[i], -bound, bound)

def sirenWeightInit(decoder):
    # weight initialization from SIREN
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        if i == 0 and ((fan_in == 2) or (fan_in == 3)): # added for 2D and 3D for cases with the learned input encoder
            bound = 1 / fan_in
            nn.init.uniform_(decoder.weights_list[i], -bound, bound)
        else:
            bound = np.sqrt(6 / fan_in) / 30
            nn.init.uniform_(decoder.weights_list[i], -bound, bound)
        bound = 0 #1 / np.sqrt(fan_in)
        nn.init.uniform_(decoder.biases_list[i], -bound, bound)

def finerWeightInit(decoder):
    # weight initialization from FINER
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        if i == 0 and ((fan_in == 2) or (fan_in == 3)): # added for 2D and 3D for cases with the learned input encoder
            bound = 1 / fan_in
            nn.init.uniform_(decoder.weights_list[i], -bound, bound)
        else:
            bound = np.sqrt(6 / fan_in) / 30
            nn.init.uniform_(decoder.weights_list[i], -bound, bound)
        # bound = 0 #1 / np.sqrt(fan_in)
        # nn.init.uniform_(decoder.biases_list[i], -bound, bound)

def sirenSameWeightInit(decoder):
    # weight initialization from SIREN, same init for all experts
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        bound = np.sqrt(6 / fan_in) / 30

        n_experts = decoder.weights_list[i].shape[0]
        for j in range(n_experts):
            if j == 0:
                w_init = torch.empty_like(decoder.weights_list[i][j]).uniform_(-bound, bound)
            decoder.weights_list[i][j] = w_init.detach()
        nn.init.uniform_(decoder.biases_list[i], 0, 0)


def SincUniformWeightInit(decoder):

    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(decoder.weights_list[i], -bound, bound)


def sirenGeomWeightInit(decoder, flip=False, r=0.5, centroids=None):
    # weight init from DiGS
    parallel = len(decoder.weights_list[0].shape) == 3
    num_experts = decoder.weights_list[0].shape[0] # k
    flip_val = -1 if flip else 1
    assert len(decoder.weights_list) >= 3, len(decoder.weights_list)
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        fan_out = decoder.weights_list[i].shape[-1]
        if i == 0:
            bound = (np.sqrt(3 / fan_out))
            nn.init.uniform_(decoder.weights_list[i], -bound, bound)
            bound = (1 / (fan_out * 1000))
            nn.init.uniform_(decoder.biases_list[i], -bound, bound)
            if centroids is not None and parallel:
                # weights: (k, in, out), bias (k, out)
                for expert_i in range(num_experts):
                    bias_term = - centroids[expert_i,None] @ decoder.weights_list[i][expert_i]
                    decoder.biases_list[i][expert_i] = decoder.biases_list[i][expert_i] + bias_term
            decoder.biases_list[i] /= 30
            decoder.weights_list[i] /= 30
        elif i == len(decoder.weights_list) - 2:
            assert fan_in == fan_out, (fan_in, fan_out)
            w = 0.5*np.pi*torch.eye(fan_out) + 0.001*torch.randn(*decoder.weights_list[i].shape)
            decoder.weights_list[i].data = w / 30
            b = 0.5 * np.pi * torch.ones(fan_out,) + 0.001 * torch.randn(*decoder.biases_list[i].shape)
            decoder.biases_list[i].data =  b / 30
        elif i == len(decoder.weights_list) - 1:
            w = -1*torch.ones(*decoder.weights_list[i].shape) + 0.00001 * torch.randn(fan_in, 1)
            decoder.weights_list[i].data = w
            b = torch.zeros(*decoder.biases_list[i].shape) + fan_in
            decoder.biases_list[i].data = b
            decoder.weights_list[i].data = flip_val * decoder.weights_list[i].data
            decoder.biases_list[i].data = flip_val * decoder.biases_list[i].data - r
        else:
            bound = (np.sqrt(3 / fan_out)) / 30
            nn.init.uniform_(decoder.weights_list[i], -bound, bound)
            bound = (1 / (fan_out * 1000)) / 30
            nn.init.uniform_(decoder.biases_list[i], -bound, bound)

def geomReluWeightInit(decoder, flip=False, r=0.5, centroids=None):
    parallel = len(decoder.weights_list[0].shape) == 3
    num_experts = decoder.weights_list[0].shape[0] # k
    flip_val = -1 if flip else 1
    assert len(decoder.weights_list) >= 3, len(decoder.weights_list)
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        fan_out = decoder.weights_list[i].shape[-1]
        if i == 0:
            decoder.weights_list[i].data.normal_(mean=0.0, std=np.sqrt(2) / np.sqrt(fan_out))
            decoder.biases_list[i].data = torch.zeros_like(decoder.biases_list[i])
            if centroids is not None and parallel:
                # weights: (k, in, out), bias (k, out)
                for expert_i in range(num_experts):
                    bias_term = - centroids[expert_i,None] @ decoder.weights_list[i][expert_i] # (1,d) @ (d,out)
                    decoder.biases_list[i][expert_i] = decoder.biases_list[i][expert_i] + bias_term.squeeze() # (out,)
        elif i == len(decoder.weights_list) - 1:
            decoder.weights_list[i].data.normal_(mean=np.sqrt(np.pi) / np.sqrt(fan_in), std=0.00001)
            decoder.biases_list[i].data = torch.zeros_like(decoder.biases_list[i])
            decoder.weights_list[i].data = flip_val * decoder.weights_list[i].data
            decoder.biases_list[i].data = decoder.biases_list[i].data - flip_val * r
        else:
            decoder.weights_list[i].data.normal_(mean=0.0, std=np.sqrt(2) / np.sqrt(fan_out))
            decoder.biases_list[i].data = torch.zeros_like(decoder.biases_list[i])


def kaimingNormalWeightInit(decoder):
    # assert len(decoder.weights_list) >= 3, len(decoder.weights_list)
    for i in range(len(decoder.weights_list)):
        nn.init.kaiming_normal_(decoder.weights_list[i], a=0.0, nonlinearity='relu', mode='fan_out')
        # decoder.biases_list[i] = 0.000001 * torch.randn_like(decoder.biases_list[i])
        # if i == len(decoder.weights_list) - 1:
        #     nn.init.xavier_normal_(decoder.weights_list[i], gain=1)
        #     decoder.biases_list[i] = 0.000001 * torch.randn_like(decoder.biases_list[i])

def kaimingUniformWeightInit(decoder):
    assert len(decoder.weights_list) >= 3, len(decoder.weights_list)
    for i in range(len(decoder.weights_list)):
        nn.init.kaiming_uniform_(decoder.weights_list[i], a=0.0, nonlinearity='relu', mode='fan_out')
        # decoder.biases_list[i] = 0.000001 * torch.randn_like(decoder.biases_list[i])
        # if i == len(decoder.weights_list) - 1:
        #     nn.init.xavier_normal_(decoder.weights_list[i], gain=1)
        #     decoder.biases_list[i] = 0.000001 * torch.randn_like(decoder.biases_list[i])

def GaussianNormalWeightInit(decoder):
    assert len(decoder.weights_list) >= 3, len(decoder.weights_list)
    for i in range(len(decoder.weights_list)):
        nn.init.uniform_(decoder.weights_list[i], -0.1, 0.1)


def UniformManagerWeightInit(decoder):
    assert len(decoder.weights_list) >= 3, len(decoder.weights_list)
    for i in range(len(decoder.weights_list)):
        if i == 0:
            # nn.init.kaiming_normal_(decoder.weights_list[i], a=0.0, nonlinearity='relu', mode='fan_out')
            decoder.weights_list[i] = 0.001 * torch.randn_like(decoder.weights_list[i])
            decoder.biases_list[i] = 1 + 0.001 * torch.randn_like(decoder.biases_list[i])
        else:
            nn.init.kaiming_normal_(decoder.weights_list[i], a=0.0, nonlinearity='relu', mode='fan_out')
            # nn.init.uniform_(decoder.weights_list[i], -0.1, 0.1)


def planarWeightInit(decoder, centroids, normals):
    parallel = len(decoder.weights_list[0].shape) == 3
    num_experts = decoder.weights_list[0].shape[0] # k
    normals = normals.to(torch.float32) if normals is not None else None
    down_scale = 1000
    assert len(decoder.weights_list) >= 3, len(decoder.weights_list)
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        fan_out = decoder.weights_list[i].shape[-1]
        if i == 0:
            nn.init.kaiming_normal_(decoder.weights_list[i], a=0.0, nonlinearity='relu', mode='fan_out')
            decoder.weights_list[i] /= down_scale
            decoder.biases_list[i] /= down_scale
            if parallel:
                for expert_i in range(num_experts):
                    decoder.weights_list[i][expert_i] += (1./fan_out)*normals[expert_i].unsqueeze(-1)
                    bias_term = -(1./fan_out)*torch.sum(centroids[expert_i] * normals[expert_i], dtype=torch.float32)
                    decoder.biases_list[i][expert_i] += bias_term
            else:
                decoder.weights_list[i] += (1./fan_out)*normals.unsqueeze(-1)
                bias_term = -(1./fan_out)*torch.sum(centroids * normals, dtype=torch.float32)
                decoder.biases_list[i] += bias_term

        else:
            nn.init.kaiming_normal_(decoder.weights_list[i], a=0.0, nonlinearity='relu', mode='fan_out')
            decoder.weights_list[i] /= down_scale
            decoder.biases_list[i] /= down_scale
            decoder.weights_list[i][...,0] += 1


def ptWeightInit(decoder, centroids, normals, pt_path):
    parallel = len(decoder.weights_list[0].shape) == 3
    num_experts = decoder.weights_list[0].shape[0] # k
    normals = normals.to(torch.float32) if normals is not None else None
    pre_trained_model = torch.load(pt_path)
    new_dict = {}
    for key, weights in pre_trained_model.items():
        if parallel:
            new_dict[key.replace('decoder.','')] = torch.stack([weights for _ in range(num_experts)], dim=0) # (k, *weights.shape)
        else:
            new_dict[key.replace('decoder.','')] = weights # # (*weights.shape,)
    decoder.load_state_dict(new_dict)
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        fan_out = decoder.weights_list[i].shape[-1]
        if i == 0:
            if parallel:
                for expert_i in range(num_experts):
                    rot_mat = torch.tensor([[normals[expert_i][1], normals[expert_i][0]],
                                            [-normals[expert_i][0], normals[expert_i][1]]], dtype=torch.float32,
                                            device=normals.device)
                    decoder.weights_list[i][expert_i] = rot_mat @ decoder.weights_list[i][expert_i]
                    bias_term = - centroids[expert_i,None] @ decoder.weights_list[i][expert_i] # (1,d) @ (d,out)
                    decoder.biases_list[i][expert_i] = decoder.biases_list[i][expert_i] + bias_term.squeeze() # (out,)
            else:
                rot_mat = torch.tensor([[normals[1], normals[0]],
                                        [-normals[0], normals[1]]], dtype=torch.float32,
                                        device=normals.device)
                decoder.weights_list[i] = rot_mat @ decoder.weights_list[i]
                bias_term = - centroids @ decoder.weights_list[i] # (1,d) @ (d,out)
                decoder.biases_list[i] = decoder.biases_list[i] + bias_term.squeeze() # (out,)



