# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import torch.nn.parallel
import torch.optim as optim

import utils.utils as utils
import yaml
import wandb
import shutil
from models import build_model

from models.stage_handler import TrainingStageHandler
from datasets import build_dataloader
from scipy.io import wavfile


def main(args):

    wandbdir = os.path.join(args.logdir, 'wandb')
    if args.logdir:
        os.makedirs(args.logdir, exist_ok=True)
        os.makedirs(wandbdir, exist_ok=True)
        model_outdir = os.path.join(args.logdir, 'trained_models')
        os.makedirs(model_outdir, exist_ok=True)
        recon_img_outdir = os.path.join(args.logdir, 'reconstructed_data')
        os.makedirs(recon_img_outdir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(args.logdir, 'config.yaml'))
        shutil.copytree('../models', os.path.join(args.logdir, 'models'), dirs_exist_ok=True)
        shutil.copy(__file__, os.path.join(args.logdir)) # backup the current training file

    cfg = yaml.safe_load(open(args.config))
    assert cfg['DATA']['n_segments'] <= cfg['MODEL']['n_experts'], "Number of segments should be smaller than or equal to the number of experts"
    # wandb_run = wandb.init(project=cfg['wandb_project'] + '_' + args.audiofile_id, entity='anu-cvml', save_code=True, dir=wandbdir)
    wandb_run = wandb.init(project=cfg['wandb_project'] + '_' + args.audiofile_id, entity='anu-cvml', save_code=True, dir=wandbdir, mode='disabled')
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

    cfg['TRAINING']['n_samples'] = cfg['TRAINING']['num_epochs']  # the training stage handler uses n_sampled (because of the 3D dataset), this hack is to allow support for RGB with minimal changes
    model_name = cfg['MODEL']['model_name']

    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get data loaders
    train_dataloader, train_set = build_dataloader(cfg, args.audiofile_id, training=True)
    cfg['MODEL']['out_dim'] = 1


    device = torch.device("cuda:" + str(args.gpu) if (torch.cuda.is_available()) else "cpu")

    SINR, _ = build_model(cfg, cfg['LOSS'])

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


    if not cfg['TRAINING']['refine_epoch'] == 0:
        refine_model_filename = os.path.join(args.logdir,
                                             '%s_model_%d.pth' % (model_name, cfg['TRAINING']['refine_epoch']))
        SINR.load_state_dict(torch.load(refine_model_filename, map_location=device))
        optimizer.step()

    if cfg['MODEL']['load_pt_manager']:
        manager_pt_checkpoint_path = os.path.join(cfg['MODEL']['manager_pt_path'])
        SINR.load_state_dict(torch.load(manager_pt_checkpoint_path))
    SINR.to(device)

    num_batches = len(train_dataloader)
    refine_flag = True

    ########################   Train the audio implicit neural representation #############################

    # copy data to GPU once since it does not change throughout training.
    data = next(iter(train_dataloader))
    coords, gt_data, segments, rate, scale = (data['coords'].to(device), data['gt_data'].to(device),
                                      data['segments'].to(device), data['rate'].to(device), data['scale'].to(device))

    # Train
    for epoch in range(cfg['TRAINING']['num_epochs']):
        if epoch <= cfg['TRAINING']['refine_epoch'] and refine_flag and not cfg['TRAINING']['refine_epoch'] == 0:
            scheduler.step()
            continue
        else:
            refine_flag = False

        for batch_idx, data in enumerate(train_dataloader):
            # save model before update
            if epoch % 100 == 0:
                utils.log_string("saving model to file :{}".format('%s_model_%d.pth' % (model_name, epoch)),
                                 log_file)
                torch.save(SINR.state_dict(),
                           os.path.join(model_outdir, '%s_model_%d.pth' % (model_name, epoch)))

            SINR.zero_grad()
            SINR.train()

            coords.requires_grad = True
            output_pred = SINR(coords)
            loss_dict = criterion(output_pred=output_pred,  coords=coords,
                                  gt={'gt_data': gt_data, 'segment': segments}, model=SINR)

            lr = torch.tensor(optimizer.param_groups[0]['lr'])
            loss_dict["lr"] = lr
            if "moe" in model_name and cfg['MODEL']['manager_q_activation'] == 'softmax' and cfg['MODEL'][
                'manager_softmax_temp_trainable']:
                loss_dict["softmax_temp"] = SINR.manager_net.q_activation.temperature.item()

            utils.log_losses_wandb(epoch, batch_idx, num_batches, loss_dict, cfg['TRAINING']['batch_size'],
                                   criterion.weight_dict)

            loss_dict["loss"].backward()
            optimizer.step()

            if epoch % 100 == 0 and cfg['TRAINING']['save_reconstructed_signals']:
                # log and save the reconstructed audio file
                if 'moe' in model_name:
                    recon_data = output_pred['selected_nonmanifold_pnts_pred'].detach().cpu().numpy().reshape(train_set.sidelength[0], train_set.sidelength[1], -1)
                else:
                    recon_data = output_pred['nonmanifold_pnts_pred'].detach().cpu().numpy().reshape(train_set.sidelength[0], train_set.sidelength[1], -1)

                wavfile.write(os.path.join(recon_img_outdir, 'reconstructed_audio_{}.png'.format(epoch)),
                              train_set.rate, recon_data)


            # save last model
            if epoch == cfg['TRAINING']['num_epochs'] - 1:
                utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f}'.format(
                    epoch, batch_idx * cfg['TRAINING']['batch_size'],
                    len(train_set), 100. * batch_idx / len(train_dataloader), loss_dict["loss"].item()), log_file)
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
    parser = argparse.ArgumentParser(description='MOE INR for audio Training')
    parser.add_argument('--config', default='../configs/config_audio.yaml', type=str, help='config file')
    parser.add_argument('--logdir', default='./log', type=str, help='path to log firectory')
    parser.add_argument('--identifier', default='debug_audio', type=str, help='unique identifier for this experiment')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index to use')
    parser.add_argument('--audiofile_id', default='gt_bach.wav', type=str, help='shape to load')
    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.identifier)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)