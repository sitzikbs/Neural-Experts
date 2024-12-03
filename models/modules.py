import torch.nn as nn
import torch.nn.functional as F
import models.initializations as inits
import torch
from collections import OrderedDict
import numpy as np

class Sine(nn.Module):
    def __init__(self, freq=30, trainable=False):
        super().__init__()
        if trainable:
            self.freq = nn.Parameter(torch.tensor(freq))
        else:
            self.freq = freq
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.freq * input)

class FINER(nn.Module):
    def __init__(self, freq=30, trainable=False):
        super().__init__()
        if trainable:
            self.freq = nn.Parameter(torch.tensor(freq))
        else:
            self.freq = freq
    def forward(self, input):
        with torch.no_grad():
            scale = torch.abs(input) + 1
        return torch.sin(self.freq * scale * input)
    
    
class SineParallel(nn.Module):
    def __init__(self, k, freq=30, trainable=False):
        super().__init__()
        self.k = k
        if trainable:
            self.freq = [nn.Parameter(torch.tensor(freq))*2**i for i in range(k)]
        else:
            self.freq = [freq*2**i for i in range(k)]
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        out = torch.cat([torch.sin(self.freq[i] * input[:, i]) for i in range(self.k)], dim=0).unsqueeze(0)
        return out


class SinePlusOne(nn.Module):
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        out = (torch.sin(30 *input) + 1) / 2
        # out = out / out.sum(-1, keepdim=True) # make sure it sums to 1
        return out

class CDFGaussian(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()

        if trainable:
            self.loc = nn.Parameter(torch.tensor(0))
            self.scale = nn.Parameter(torch.tensor(0.03))
        else:
            self.loc = 0.0
            self.scale = 0.03
    def forward(self, input):
        out = 0.5 * (1 + torch.erf((input - self.loc) * (1/self.scale) / np.sqrt(2)))
        out = out / torch.clamp(out.sum(-1, keepdim=True), 1e-6) # make sure it sums to 1
        return out

class gaussian(nn.Module):
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std
    def forward(self, input):
        # beyond periodicity paper https://github.com/samgregoost/Beyond_periodicity/blob/c5506a3d906e2c3e3b1df1bde0c5029f687e7d84/run_nerf_helpers.py#L97
        return (-0.5*(input)**2/self.std**2).exp()


class SoftPlusSumToOne(nn.Module):
    def forward(self, input):
        out = F.softplus(input)
        out = out / out.sum(-1, keepdim=True) # make sure it sums to 1
        return out

class QuadraticActivation(nn.Module):
    # from https://github.com/kwea123/Coordinate-MLPs
    def __init__(self, a=1., trainable=False):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)



class SineParallel(nn.Module):
    def __init__(self, k, freq=30, trainable=False):
        super().__init__()
        self.k = k
        if trainable:
            self.freq = [nn.Parameter(torch.tensor(freq))*2**i for i in range(k)]
        else:
            self.freq = [freq*2**i for i in range(k)]
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        out = torch.cat([torch.sin(self.freq[i] * input[:, i]) for i in range(self.k)], dim=0).unsqueeze(0)
        return out


class SinePlusOne(nn.Module):
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        out = (torch.sin(30 *input) + 1) / 2
        # out = out / out.sum(-1, keepdim=True) # make sure it sums to 1
        return out

class CDFGaussian(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()

        if trainable:
            self.loc = nn.Parameter(torch.tensor(0))
            self.scale = nn.Parameter(torch.tensor(0.03))
        else:
            self.loc = 0.0
            self.scale = 0.03
    def forward(self, input):
        out = 0.5 * (1 + torch.erf((input - self.loc) * (1/self.scale) / np.sqrt(2)))
        out = out / torch.clamp(out.sum(-1, keepdim=True), 1e-6) # make sure it sums to 1
        return out

class gaussian(nn.Module):
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std
    def forward(self, input):
        # beyond periodicity paper https://github.com/samgregoost/Beyond_periodicity/blob/c5506a3d906e2c3e3b1df1bde0c5029f687e7d84/run_nerf_helpers.py#L97
        return (-0.5*(input)**2/self.std**2).exp()


class SoftPlusSumToOne(nn.Module):
    def forward(self, input):
        out = F.softplus(input)
        out = out / out.sum(-1, keepdim=True) # make sure it sums to 1
        return out

class QuadraticActivation(nn.Module):
    # from https://github.com/kwea123/Coordinate-MLPs
    def __init__(self, a=1., trainable=False):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)


class FullyConnectedNN(nn.Module):
    '''A fully connected neural network.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', init_type='siren',
                 input_encoding=None,
                 sphere_init_params=[1.6,1.0], verbose=True, init_r=0.5, freq=30, trainable_freqs=False, module_name=''):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_hidden_layers = num_hidden_layers
        self.hidden_features = hidden_features
        self.outermost_linear = outermost_linear
        self.init_type = init_type
        self.input_encoding = input_encoding
        self.sphere_init_params = sphere_init_params
        first_layer_dim = in_features

        self.weights_list = nn.ParameterList([])
        self.biases_list = nn.ParameterList([])
        self.c_list = nn.ParameterList([])
        self.weights_list.append(nn.Parameter(torch.zeros(first_layer_dim, hidden_features)))
        self.biases_list.append(nn.Parameter(torch.zeros(hidden_features)))
        for i in range(num_hidden_layers):
            self.weights_list.append(nn.Parameter(torch.zeros(hidden_features, hidden_features)))
            self.biases_list.append(nn.Parameter(torch.zeros(hidden_features)))
        self.weights_list.append(nn.Parameter(torch.zeros(hidden_features, out_features)))
        self.biases_list.append(nn.Parameter(torch.zeros(out_features)))


        self.module_name = module_name

        nl_dict = {'sine': Sine(freq, trainable_freqs), 'relu': nn.ReLU(inplace=True), 'softplus': nn.Softplus(beta=100),
                    'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'finer': FINER()}
        assert nonlinearity in nl_dict.keys()
        self.nl = nl_dict[nonlinearity]

        init_dict = {'siren': lambda: inits.sirenWeightInit(self), 
                     'finer': lambda: inits.finerWeightInit(self),
                     'geometric_sine': lambda: inits.sirenGeomWeightInit(self, flip=False, r=init_r),
                     'geometric_relu': lambda: inits.geomReluWeightInit(self, flip=False, r=init_r),
                     'normal': lambda: inits.kaimingNormalWeightInit(self),
                     'kaiminguniform': lambda: inits.kaimingUniformWeightInit(self),
                        }
        init_dict[init_type]()
    
    def forward(self, input):
        # coords: (1,n,d)

        # Run through network
        x = torch.einsum('...nd,dh->...nh', input, self.weights_list[0]) + self.biases_list[0] # (1,n,h)
        x = self.nl(x)
        for i in range(self.num_hidden_layers):
            x = torch.einsum('...nd,dh->...nh', x, self.weights_list[i+1]) + self.biases_list[i+1] # (1,n,h)
            x = self.nl(x)

        x = torch.einsum('...nh,ho->...no', x, self.weights_list[-1]) + self.biases_list[-1] # (1,n,o)

        if not self.outermost_linear:
            x = self.nl(x)

        # Apply output scaling if any
        if self.init_type == 'mfgi' or self.init_type == 'geometric_sine':
            radius, scaling = self.sphere_init_params
            x = torch.sign(x)*torch.sqrt(x.abs()+1e-8)
            x -= radius # 1.6
            x *= scaling # 1.0

        return x


class ParallelFullyConnectedNN(nn.Module):
    '''K parallel connected neural networks.
    '''

    def __init__(self, k, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', init_type='siren',
                 input_encoding=None,
                 sphere_init_params=[1.6,1.0], init_r=0.5, freq=30,
                 module_name=''):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_hidden_layers = num_hidden_layers
        self.hidden_features = hidden_features
        self.outermost_linear = outermost_linear
        self.init_type = init_type
        self.input_encoding = input_encoding
        self.sphere_init_params = sphere_init_params

        first_layer_dim = in_features


        self.weights_list = nn.ParameterList([])
        self.biases_list = nn.ParameterList([])
        self.weights_list.append(nn.Parameter(torch.zeros(k, first_layer_dim, hidden_features)))
        self.biases_list.append(nn.Parameter(torch.zeros(k, hidden_features)))
        for _ in range(num_hidden_layers):
            self.weights_list.append(nn.Parameter(torch.zeros(k, hidden_features, hidden_features)))
            self.biases_list.append(nn.Parameter(torch.zeros(k, hidden_features)))
        self.weights_list.append(nn.Parameter(torch.zeros(k, hidden_features, out_features)))
        self.biases_list.append(nn.Parameter(torch.zeros(k, out_features)))

        self.module_name = module_name

        nl_dict = {'sine': Sine(), 'relu': nn.ReLU(inplace=True), 'softplus': nn.Softplus(beta=100),
                    'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(),'finer': FINER()}
        assert nonlinearity in nl_dict.keys(), nonlinearity
        self.nl = nl_dict[nonlinearity]

        init_dict = {'siren': lambda: inits.sirenWeightInit(self),
                     'finer': lambda: inits.finerWeightInit(self),
                     'sirensame': lambda: inits.sirenSameWeightInit(self),
                     'geometric_sine': lambda: inits.sirenGeomWeightInit(self, flip=False, r=init_r),
                     'geometric_relu': lambda: inits.geomReluWeightInit(self, flip=False, r=init_r),
                     'normal': lambda: inits.kaimingNormalWeightInit(self),
                     'kaiminguniform': lambda: inits.kaimingUniformWeightInit(self),
                    }
        with torch.no_grad():
            init_dict[init_type]()

    def forward(self, x):
        # coords: (...,n,d)

        # Run through network
        x = torch.einsum('...nd,kdh->...knh', x, self.weights_list[0]) + self.biases_list[0].unsqueeze(-2) # (1,k,n,h)
        x = self.nl(x)
        for i in range(self.num_hidden_layers):
            x = torch.einsum('...knd,kdh->...knh', x, self.weights_list[i+1]) + self.biases_list[i+1].unsqueeze(-2) # (1,k,n,h)
            x = self.nl(x)
        x = torch.einsum('...knh,kho->...kno', x, self.weights_list[-1]) + self.biases_list[-1].unsqueeze(-2) # (1,k,n,o)
        if not self.outermost_linear:
            x = self.nl(x)

        # Apply output scaling if any
        if self.init_type == 'mfgi' or self.init_type == 'geometric_sine':
            radius, scaling = self.sphere_init_params
            x = torch.sign(x)*torch.sqrt(x.abs()+1e-8)
            x -= radius # 1.6
            x *= scaling # 1.0

        # return x[...,0,:,:] # (...,n,1), taking the 1st expert's output
        return x # (...,k,n,1)


class InputEncoder(nn.Module):
    def __init__(self, cfg, input_encoding, hidden_dim, module_name=''):
        super().__init__()
        self.input_encoding = input_encoding
        hidden_features = hidden_dim
        in_features = cfg['in_dim']
        self.first_layer_dim = in_features

        # concatenated input features
        if 'FF' in self.input_encoding:
            # Fourier Features
            assert hidden_features % 2 == 0
            self.bvals_size = hidden_features // 2
            bvals = torch.randn(size=[self.bvals_size, cfg['in_dim']], dtype=torch.float32) * 1
            self.register_buffer("bvals", bvals)
            self.first_layer_dim = hidden_features + cfg['in_dim']
        elif 'PE' in self.input_encoding:
            # Nerf-style Positional Ecoding from NerfStudio
            bvals = 2 ** torch.linspace(0.0, 5.0, 6)
            self.register_buffer("bvals", bvals)
            self.first_layer_dim = cfg['in_dim'] * 6 * 2 + cfg['in_dim']
        elif 'dino' in self.input_encoding:
            self.first_layer_dim = cfg['in_dim'] + cfg['dino_dim']

        # learned input encoding
        if 'learned' in self.input_encoding:
            parsed_str = self.input_encoding.split('_')
            enc_hidden_features = int(parsed_str[1])
            enc_n_layers = int(parsed_str[2])
            nl = parsed_str[3]
            init = parsed_str[4]
            self.encoder = FullyConnectedNN(self.first_layer_dim, enc_hidden_features, num_hidden_layers=enc_n_layers,
                                    hidden_features=enc_hidden_features, outermost_linear=False,
                                    nonlinearity=nl, init_type=init,
                                    module_name=module_name + '.encoder')
            self.first_layer_dim = enc_hidden_features + self.first_layer_dim if 'cat' in self.input_encoding else enc_hidden_features


    def forward(self, coords, **kwargs):
        # Apply input encoding if any
        if 'FF' in self.input_encoding:
            x = (2*np.pi*coords) @ self.bvals.T # (1, n, bvals_size)
            x = torch.cat([torch.sin(x), torch.cos(x)], axis=-1) / np.sqrt(self.bvals_size) # (1, n, bvals_size*2)
            x = torch.cat([coords, x], axis=-1) # (1, n, bvals_size*2 + d)
        elif 'PE' in self.input_encoding:
            x = coords[..., None] * self.bvals  # (1,n,d,num_scales)
            x = x.reshape(*x.shape[:-2], -1)  # (1,n,d*num_scales)
            x = torch.sin(torch.cat([x, x + np.pi / 2.0], dim=-1)) # (1,n,2*d*num_scales)
            x = torch.cat([coords, x], axis=-1)  # (1,n,d+2*d*num_scales)
        elif 'dino' in self.input_encoding:
            dino = kwargs['dino']
            x = torch.cat([coords, dino], axis=-1)  # (1,n,d+2*d*num_scales)
        else:
            x = coords

        if 'learned' in self.input_encoding:
            x = self.encoder(x)
            if 'cat' in self.input_encoding:
                x[0] = torch.cat([coords, x[0]], axis=-1)
        
        return x

class tSoftMax(nn.Module):
    def __init__(self, temperature, dim=-1, trainable=False):
        super().__init__()
        if trainable:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature
        self.dim = dim
        self.activation = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.activation(x / self.temperature)

class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

