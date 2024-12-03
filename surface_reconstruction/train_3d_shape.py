# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os
import sys
import time
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import utils.utils as utils
import yaml
import wandb
import shutil
from models import build_model
from models.stage_handler import TrainingStageHandler
from datasets import build_dataloader

from sdf_utils import dict2device, compute_full_grid, pred_sdf_to_mesh, grid_pred2metrics, mesh2metrics, pred_sdf_to_coloured_mesh
import utils.utils as utils


def lossdict2str(loss_dict):
    string = ""
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            val = v.item()
        else:
            val = v
        if val == 0.0:
            continue
        if k == 'lr' or val < 1e-4:
            string += f'{k}: {val:.4e}, '
        else:
            string += f'{k}: {val:.8f}, '
    return string

def main(args):
    t0 = time.time()

    wandbdir = os.path.join(args.logdir, 'wandb')
    if args.logdir:
        os.makedirs(args.logdir, exist_ok=True)
        os.makedirs(wandbdir, exist_ok=True)
        model_outdir = os.path.join(args.logdir, 'trained_models')
        os.makedirs(model_outdir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(args.logdir, 'config.yaml'))
        shutil.copytree('../models', os.path.join(args.logdir, 'models'), dirs_exist_ok=True)
        shutil.copy(__file__, os.path.join(args.logdir)) # backup the current training file

    cfg = yaml.safe_load(open(args.config))
    # wandb_run = wandb.init(project=cfg['wandb_project'] + '_' + args.shape_id, entity='anu-cvml', save_code=True, dir=wandbdir)
    wandb_run = wandb.init(project=cfg['wandb_project'] + '_' + args.shape_id, entity='anu-cvml', save_code=True, dir=wandbdir, mode="disabled")
    cfg['WANDB'] = {'id': wandb_run.id, 'project': wandb_run.project, 'entity': wandb_run.entity}
    wandb_run.name = args.identifier
    wandb.config.update(cfg)  # adds all the arguments as config variables
    wandb.run.log_code(".")
    # define our custom x axis metric
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    log_filename = os.path.join(args.logdir, 'out.log')
    log_file = open(log_filename, 'w')
    print(args)
    print("torch version: ", torch.__version__)
    device = torch.device("cuda:" + str(args.gpu) if (torch.cuda.is_available()) else "cpu")
    cfg['device'] = device

    cfg['TRAINING']['n_samples'] = cfg['TRAINING']['num_epochs']  # the training stage handler uses n_sampled (because of the 3D dataset), this hack is to allow support for RGB with minimal changes
    model_name = cfg['MODEL']['model_name']
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get data loaders
    train_dataloader, train_set = build_dataloader(cfg, args.shape_id, training=True)
    cfg['MODEL']['out_dim'] = 1
    print(f'Finished building dataset and dataloader: {time.time()-t0:.2f}s'); t0 = time.time()

    SINR, _ = build_model(cfg, cfg['LOSS'])
    print(f'Finished building model: {time.time()-t0:.2f}s'); t0 = time.time()

    n_parameters = utils.count_parameters(SINR)
    wandb.log({"number of paramters": n_parameters})
    utils.log_string("Number of parameters in the current model:{}".format(n_parameters), log_file)

    # Setup Adam optimizers
    training_stage_handler = TrainingStageHandler(cfg['TRAINING']['stages'], SINR, cfg)
    criterion = training_stage_handler.criterion
    lr = cfg['TRAINING']['lr'] if isinstance(cfg['TRAINING']['lr'], float) else cfg['TRAINING']['lr']['all']
    if "moe" in model_name:
        optimizer = optim.Adam(training_stage_handler.get_trainable_params(),
                               lr=lr, betas=(0.9, 0.999))
        training_stage_handler.freeze_params()
    else:
        optimizer = optim.Adam(training_stage_handler.get_trainable_params(),
                               lr=lr, betas=(0.9, 0.999))

    scheduler = training_stage_handler.get_scheduler(optimizer)
    print(f'Finished building optimizer and scheduler: {time.time()-t0:.2f}s'); t0 = time.time()


    if not cfg['TRAINING']['refine_epoch'] == 0:
        refine_model_filename = os.path.join(args.logdir,
                                             '%s_model_%d.pth' % (model_name, cfg['TRAINING']['refine_epoch']))
        SINR.load_state_dict(torch.load(refine_model_filename, map_location=device))
        optimizer.step()

    if cfg['MODEL']['load_pt_manager']:
        manager_pt_checkpoint_path = os.path.join(cfg['MODEL']['manager_pt_path'])
        SINR.load_state_dict(torch.load(manager_pt_checkpoint_path))
        print(f'Loading pretrained model from {manager_pt_checkpoint_path}')
    SINR.to(device)

    num_batches = len(train_dataloader)
    refine_flag = True
    print(f'Finished all setup: {time.time()-t0:.2f}s'); t0 = time.time()
    # import pdb; pdb.set_trace()

    ########################   Train the shape implicit neural representation #############################

    epoch_t0 = time.time()
    for epoch, data in enumerate(train_dataloader):
        if epoch > cfg['TRAINING']['num_epochs']:
            break
        if epoch <= cfg['TRAINING']['refine_epoch'] and refine_flag and not cfg['TRAINING']['refine_epoch'] == 0:
            scheduler.step()
            continue
        else:
            refine_flag = False
        
        if epoch % 100 == 0:
            # utils.log_string("saving model to file :{}".format('%s_model_%d.pth' % (model_name, epoch)),
            #                  log_file)
            torch.save(SINR.state_dict(),
                        os.path.join(model_outdir, '%s_model_%d.pth' % (model_name, epoch)))

        data = dict2device(data, device, non_blocking=True)

        metrics_dict = {}
        output_pred = SINR(data['nonmnfld_points'], data['mnfld_points'])

        log_every = 500
        if epoch % log_every == 0:
            time_since_last_logging = time.time() - epoch_t0
            metrics_dict['time'] = time_since_last_logging
            with torch.no_grad():
                # t0 = time.time()
                grid_res = train_set.grid_res
                grid_pnts = train_set.grid_points
                grid_sdfs_gt = train_set.grid_sdfs

                grid_pred = compute_full_grid(grid_pnts, device, SINR, process_size=256*256*2)
                metrics_dict = grid_pred2metrics(grid_pred, grid_sdfs_gt, grid_res, train_set, device, metrics_dict=metrics_dict)

                if not cfg['TRAINING']['segmentation_mode']:
                    mesh_dir = os.path.join(args.logdir, 'meshes')
                    os.makedirs(mesh_dir, exist_ok=True)
                    mesh_path = os.path.join(mesh_dir, f'mesh_{epoch:05d}.ply')
                    if 'moe' in cfg['MODEL']['model_name']:
                        mesh = pred_sdf_to_coloured_mesh(grid_pred['nonmanifold_pnts_pred'], mesh_path, grid_res, grid_pnts, 
                                                         train_set, ".", device, SINR, process_size=256*256*2)
                    else:
                        mesh = pred_sdf_to_mesh(grid_pred['nonmanifold_pnts_pred'], mesh_path, grid_res, grid_pnts, train_set, ".")
                    metrics_dict = mesh2metrics(mesh, train_set, device, metrics_dict=metrics_dict)
            epoch_t0 = time.time()
        
        output_pred['epoch'] = epoch
        output_pred['logdir'] = args.logdir
        loss_dict = criterion(output_pred=output_pred, data = data,
                                    dataset=train_set)
        loss_dict.update(metrics_dict)

        lr = torch.tensor(optimizer.param_groups[0]['lr'])
        loss_dict["lr"] = lr
        if "moe" in model_name and cfg['MODEL']['manager_q_activation'] == 'softmax' and cfg['MODEL'][
            'manager_softmax_temp_trainable']:
            loss_dict["softmax_temp"] = SINR.manager_net.q_activation.temperature.item()

        utils.log_losses_wandb(epoch, -1, 1, loss_dict, 1, criterion.weight_dict)
        if epoch % log_every == 0:
            utils.log_string(f'{epoch:05d} ' + lossdict2str(loss_dict), log_file)

        SINR.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

        # save last model
        if epoch == cfg['TRAINING']['num_epochs'] - 1:
            utils.log_string('Epoch: {} Loss: {:.5f}'.format(
                epoch, loss_dict["loss"].item()), log_file)
            utils.log_string("saving model to file :{}".format('%s_model_%d.pth' % (model_name, epoch)),
                                log_file)
            torch.save(SINR.state_dict(),
                        os.path.join(model_outdir, '%s_model_%d.pth' % (model_name, epoch)))

        if epoch > training_stage_handler.get_end_iteration():
            print("Moved to the next training stage...")
            training_stage_handler.move_to_the_next_training_stage(optimizer, scheduler)
            criterion = training_stage_handler.criterion
        scheduler.step()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DiGS MOE 2D Training')
    parser.add_argument('--config', default='../configs/config_2D.yaml', type=str, help='config file')
    parser.add_argument('--logdir', default='./log', type=str, help='path to log firectory')
    parser.add_argument('--identifier', default='debug', type=str, help='unique identifier for this experiment')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index to use')
    parser.add_argument('--shape_id', default='L', type=str, help='shape to load')
    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.identifier)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)