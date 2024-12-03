# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# This file contains the code for handeling different training stages as a class

from itertools import zip_longest, tee
from collections import defaultdict
from .sdf_losses import SDFShapeLoss
from .rgb_losses import RGBImageLoss
from .audio_losses import AudioLoss
import torch.optim.lr_scheduler as lr_scheduler

class TrainingStageHandler():
    def __init__(self, stages_list, model, cfg):
        self.cfg = cfg
        self.n_samples = cfg['TRAINING']['n_samples']
        self.stages_list = stages_list
        self.current_stage_idx = 0
        self.model = model
        self.current_stage = self.stages_list[self.current_stage_idx]

        if isinstance(cfg['TRAINING']['lr'], dict):
            self.lr_dict = cfg['TRAINING']['lr']
        elif isinstance(cfg['TRAINING']['lr'], float):
            self.lr_dict = {'all': cfg['TRAINING']['lr'], 'experts': cfg['TRAINING']['lr'],
                            'manager': cfg['TRAINING']['lr'], 'experts_encoder': cfg['TRAINING']['lr'],
                            'full_experts': cfg['TRAINING']['lr']}
        else:
            raise ValueError("lr must be either float or dict")

        if self.current_stage['params'] == 'intermitent':
            self.stages_list = self.generate_intermitent_stages(self.stages_list)
            self.current_stage = self.stages_list[self.current_stage_idx]

        self.default_criterion_type = cfg['LOSS']['loss_type']
        if 'rgb' in cfg['MODEL']['model_name']:
            self.default_criterion = RGBImageLoss(cfg=cfg['LOSS'], model_name=self.cfg['MODEL']['model_name'],
                                                  model=self.model, n_experts=cfg['MODEL']['n_experts'])
        else:
            self.default_criterion = SDFShapeLoss(cfg=cfg['LOSS'], model_name=self.cfg['MODEL']['model_name'],
                                              model=self.model, n_experts=cfg['MODEL']['n_experts'])
        self.get_criterion()
        self.cfg = cfg

        if 'moe' in cfg['MODEL']['model_name']:
            self.param_dict = {'experts': [model.decoder.parameters],
                               'manager': [model.manager_net.parameters]}
        else:
            self.param_dict = {'experts': [model.decoder.parameters] }

        # if cfg['MODEL']['use_hashtable']:
        #     hashtable_params = [model.hashtable.parameters]
        #     self.param_dict['experts'].extend(hashtable_params)


        if cfg['MODEL']['manager_conditioning'] == 'CNN' or cfg['MODEL']['manager_conditioning'] == 'FCN' or cfg['MODEL']['manager_conditioning'] == 'expert_weights':
            if 'moe' in cfg['MODEL']['model_name']:
                self.param_dict['manager'].append(model.manager_conditioner.cond_encoding.parameters)

        if 'learned' in cfg['MODEL']['decoder_input_encoding']: # add encoder params
            self.param_dict['experts_encoder'] = [model.decoder_input_encoding_module.parameters]
            self.param_dict['full_experts'] = [model.decoder.parameters, model.decoder_input_encoding_module.parameters]
        else:
            self.param_dict['full_experts'] = []
            self.param_dict['experts_encoder'] = []


        if 'learned' in cfg['MODEL']['manager_input_encoding'] and 'moe' in cfg['MODEL']['model_name']:
            # self.param_dict['manager_encoder'] = [model.manager_input_encoding_module.parameters]
            self.param_dict['manager'].append(model.manager_input_encoding_module.parameters)

        all_params = []
        for key in self.param_dict.keys():
            if not 'encoder' in key and not key == 'experts':
                all_params.extend([[item] for item in self.param_dict[key]])
            # all_params = [[item] for sublist in list(self.param_dict.values()) for item in sublist]
        self.param_dict['all'] = all_params


    def generate_intermitent_stages(self, stages_list):
        # fenerate intermitent stages according to the last two entries of stages_list
        # the last two entries are the start and end of the intermitent stage

        new_stage_list = []
        stages_list.pop(0)
        n_substages = len(stages_list)
        step_size = 0
        for i in range (n_substages):
            step_size = step_size + stages_list[i]['end_iteration_frac']
        n_stages = int(1. / step_size)
        end_iteration_frac = 0
        for i in range(n_stages+1): # +1 to make sure we reach 1.0
            for j in range(n_substages):
                end_iteration_frac = end_iteration_frac + stages_list[j]['end_iteration_frac']
                new_stage_list.append({'end_iteration_frac': end_iteration_frac,
                                       'params': stages_list[j]['params'],
                                       'loss_type': stages_list[j]['loss_type']})

        return new_stage_list

    def get_criterion(self):
        # get the criterion for the current stage
        self.cfg['LOSS']['loss_type'] = self.current_stage['loss_type']
        if 'loss_type' in self.current_stage.keys():
            if 'rgb' in self.cfg['MODEL']['model_name']:
                self.criterion = RGBImageLoss(cfg=self.cfg['LOSS'], model_name=self.cfg['MODEL']['model_name'],
                                            model=self.model, n_experts=self.cfg['MODEL']['n_experts'])
            elif 'audio' in self.cfg['MODEL']['model_name']:
                self.criterion = AudioLoss(cfg=self.cfg['LOSS'], model_name=self.cfg['MODEL']['model_name'],
                                            model=self.model, n_experts=self.cfg['MODEL']['n_experts'])
            else:
                self.criterion = SDFShapeLoss(cfg=self.cfg['LOSS'], model_name=self.cfg['MODEL']['model_name'],
                                            model=self.model, n_experts=self.cfg['MODEL']['n_experts'])

        else:
            self.cfg['LOSS']['loss_type'] = self.default_criterion_type
            self.criterion = self.default_criterion

        print('Currently optimizing for {}'.format(self.cfg['LOSS']['loss_type']))

    def get_trainable_params(self):
        out_list = []
        lr = self.lr_dict[self.current_stage['params']]
        for i, params in enumerate(self.param_dict[self.current_stage['params']]):
            if isinstance(params, list):
                out_list.append({'params': params[0](), 'lr': lr})
            else:
                out_list.append({'params': params(), 'lr': lr})
        return out_list

    def get_end_iteration(self):
        return self.current_stage['end_iteration_frac'] * self.n_samples

    def get_frozen_params_dict(self):
        out_list = []
        for params in self.param_dict['all']:
            # check if current params is list
            if isinstance(self.param_dict[self.current_stage['params']][0], list):
                if not params in self.param_dict[self.current_stage['params']]:
                    out_list.append(params[0])
            else:
                if not params[0] in self.param_dict[self.current_stage['params']]:
                    out_list.append(params[0])
        return out_list

    def move_to_the_next_training_stage(self, optimizer, scheduler):
        # move to the next training stage
        # replaces the parameters in the optimizer with the new ones and sets the requires_grad to True/False
        self.current_stage_idx += 1
        if self.current_stage_idx >= len(self.stages_list):
            raise Warning("No more training stages, using the last stage")
        self.current_stage = self.stages_list[self.current_stage_idx]
        current_lr = optimizer.param_groups[0]['lr'] #TODO add support for different lr for each parameter group, this assumes they are all the same, the problem is mapping the parameter groups, may require refactor

        optimizer.param_groups = []
        optimizer.state = defaultdict(dict)
        for param_dict in self.get_trainable_params():
            params = list(param_dict['params'])
            optimizer.add_param_group({'params': params, 'lr': current_lr})
            for param in params:
                param.requires_grad = True
        self.freeze_params()
        self.get_criterion()

    def freeze_params(self):
        for param_gen in self.get_frozen_params_dict():
            for param in param_gen():
                param.requires_grad = False

    def get_scheduler(self, optimizer):
        if self.cfg['TRAINING']['lr_scheduler'] == 'ExponentialLR':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg['TRAINING']['lr_gamma'])  # Does nothing
        else:  # no scheduler
            scheduler = lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        return scheduler


if __name__ == '__main__':
    # test the stage handler in a basic example
    import yaml
    import torch
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(3, 3)
            self.manager_net = torch.nn.Linear(3, 3)

    cfg = yaml.safe_load(open('../configs/config_2D.yaml'))
    model = Model()
    stage_handler = TrainingStageHandler(cfg['TRAINING']['stages'], model, cfg)

    optimizer = torch.optim.Adam(stage_handler.get_trainable_params(),
                           lr=cfg['TRAINING']['lr'], betas=(0.9, 0.999))
    stage_handler.freeze_params()

    print(cfg['TRAINING']['stages'][0])
    print("Optimizer parameter groups: {}".format(optimizer.param_groups))
    print("Manager network requires grad: {}".format(model.manager_net.weight.requires_grad))
    print("Decoder requires grad: {}".format(model.decoder.weight.requires_grad))
    print("Moving to the next stage")

    for i in range(1, len(cfg['TRAINING']['stages'])):
        stage_handler.move_to_the_next_training_stage(optimizer)
        print(cfg['TRAINING']['stages'][i])
        print("Optimizer parameter groups: {}".format(optimizer.param_groups))
        print("Manager network requires grad: {}".format(model.manager_net.weight.requires_grad))
        print("Decoder requires grad: {}".format(model.decoder.weight.requires_grad))
        print("Moving to the next stage")


