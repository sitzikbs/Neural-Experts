# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import utils.visualizations as vis
import numpy as np
import torch.nn.parallel
import utils.utils as utils
import importlib
import yaml
from PIL import Image
from datasets import build_dataloader
# import postprocess_outputs
from models.audio_losses import AudioReconLossSingle
import pickle
from scipy.io import wavfile

def main(args):
    # get training and testing parameters
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # get data loaders
    test_dataloader, test_set = build_dataloader(cfg, args.audiofile_id, training=False)
    cfg['MODEL']['out_dim'] = 1


    # get model
    device = torch.device("cuda:" + str(args.gpu) if (torch.cuda.is_available()) else "cpu")

    spec = importlib.util.spec_from_file_location('build_model_from_logdir', os.path.join(args.logdir, 'models', '__init__.py'))
    build_model_from_logdir = importlib.util.module_from_spec(spec)
    sys.modules['build_model_from_logdir'] = build_model_from_logdir
    spec.loader.exec_module(build_model_from_logdir)
    SINR, criterion = build_model_from_logdir.build_model_from_logdir(args.logdir, cfg, cfg['LOSS']).get()
    # SINR.to(device)

    model_dir = os.path.join(args.logdir, 'trained_models')
    output_dir = os.path.join(args.logdir, 'audio_recon')

    os.makedirs(output_dir, exist_ok=True)

    # get loss
    _, test_data = next(enumerate(test_dataloader))

    coords, gt_data = test_data['coords'].to(device), test_data['gt_data'].to(device)
    gt_data = gt_data.reshape(test_set.sidelength, -1)

    epoch_n_eval = cfg['TESTING']['epoch_n_eval']
    eval_epochs = range(epoch_n_eval[0], epoch_n_eval[1], epoch_n_eval[2])


    mse_module = AudioReconLossSingle()
    eval_metrics = {'Epochs': [], 'MSE': []}

    for epoch in eval_epochs:
        print("Visualizing  {} epoch {}".format(args.audiofile_id, epoch))

        model_filename = os.path.join(model_dir, '%s_model_%d.pth' % (cfg['MODEL']['model_name'], epoch))
        SINR.load_state_dict(torch.load(model_filename, map_location=device))
        SINR.to(device)
        SINR.eval()

        # coords.requires_grad_()
        output_pred = SINR(coords)

        audio_dict = {}
        img_dict = {}
        if "moe" in cfg['MODEL']['model_name']:
            recon_data = output_pred['selected_nonmanifold_pnts_pred'].reshape(test_set.sidelength, -1)
            selected_expert_idx = output_pred['nonmnfld_selected_expert_idx']
            q = output_pred['nonmnfld_q']
            if cfg['TESTING'].get('plot_q_grad', True):
                q_grad = utils.experts_gradient(coords, q).norm(2, dim=-1)
                q_grad = q_grad.gather(dim=1, index=selected_expert_idx[None, None, :]).cpu().detach().numpy().reshape(
                    test_set.sidelength[0], test_set.sidelength[1])
            else:
                q_grad = None
            selected_expert_idx = selected_expert_idx.reshape(test_set.sidelength, -1)
            q = q.squeeze().reshape(-1, test_set.sidelength)

            per_expert_recon_data = output_pred['nonmanifold_pnts_pred']

            img_dict['recon_waveform'], img_dict['waveform_experts'] = (
                vis.plot_audio_experts(coords.reshape(test_set.sidelength, -1), recon_data, gt_data, selected_expert_idx,
                                                                   q, q_grad, per_expert_recon_data))
            del q, per_expert_recon_data, # clear some memory

        else:
            recon_data = output_pred['nonmanifold_pnts_pred'].permute(0, 2, 1).reshape(test_set.sidelength, -1)
            img_dict['recon_waveform'] = vis.plot_waveforms(coords.reshape(test_set.sidelength, -1), recon_data, gt_data)

        del output_pred # clear some memory

        eval_metrics['Epochs'].append(epoch)
        with torch.no_grad():
            audio_dict['reconstructed_data'] = recon_data
            mse = mse_module.compute_loss(recon_data,  gt_data)
            eval_metrics['MSE'].append(mse.detach().item())
            eval_metric_img_dict = vis.plot_eval_metrics(eval_metrics,
                                                         data_range={'x':[0, max(eval_epochs)],'y':[0, 0.03]})
            # psnr = psnr_module(gt_img.reshape(test_set.sidelength[0], test_set.sidelength[1], -1), mse)
            # eval_metrics['PSNR'].append(psnr.item())

            img_dict.update(eval_metric_img_dict)

            # save the generated images
            for key, val in img_dict.items():
                print('Saving image: ', key)
                if val is not None:
                    if type(val) is list:
                        os.makedirs(os.path.join(output_dir, key), exist_ok=True)
                        for i, v in enumerate(val):
                            im = Image.fromarray(v)
                            im.save(os.path.join(output_dir, key, "expert_" + str(i) + "_" + str(epoch).zfill(6) + ".png"))
                    else:
                        im = Image.fromarray(val)
                        im.save(os.path.join(output_dir, key + "_" + str(epoch).zfill(6) + ".png"))

            for key, val in audio_dict.items():
                print('Saving audio: ', key)
                if val is not None:
                    if type(val) is list:
                        os.makedirs(os.path.join(output_dir, key), exist_ok=True)
                        for i, v in enumerate(val):
                            wavfile.write(os.path.join(output_dir, key, "expert_" + str(i) + "_" + str(epoch).zfill(6) + ".wav"),
                                          test_set.rate, v.squeeze().detach().cpu().numpy())
                    else:
                        wavfile.write(os.path.join(output_dir, key + "_" + str(epoch).zfill(6) + ".wav"),
                                      test_set.rate, val.squeeze().detach().cpu().numpy())

    # save the evaluation metrics to a file
    eval_metrics_filename = os.path.join(args.logdir, 'eval_metrics.pickle')
    with open(eval_metrics_filename, 'wb') as f:
        pickle.dump(eval_metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(eval_metrics)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Testing RGB MoE INR')
    parser.add_argument('--config', default='../configs/config_audio.yaml', type=str, help='config file')
    # parser.add_argument('--logdir', default='./log/debug_audio', type=str, help='path to log firectory')
    parser.add_argument('--logdir', default='./log/sota_audio/Our_SIREN_MoE_gt_counting.wav', type=str, help='path to log firectory')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index to use')
    parser.add_argument('--audiofile_id', default='gt_counting.wav', type=str, help='shape to load')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # postprocess_outputs.postprocess_outputs_join(args.logdir)

    # import matplotlib.pyplot as plt
    #
    # plt.imshow(recon_img)
    # plt.show()


