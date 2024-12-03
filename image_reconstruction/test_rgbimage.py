# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import utils.visualizations as vis
import utils.diff_operators as diff_operators
import utils.dataio as dataio
import utils.utils as utils
import numpy as np
import torch.nn.parallel
import importlib
import yaml
from PIL import Image
from datasets import build_dataloader
from models.rgb_losses import PSNR, RGBReconLossSingle
import pickle
import cmapy
import cv2


def convert_to_uint8(img):
    if img.shape[-1] == 1:
        img = img.squeeze()
        img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    else:
        img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    return img


def main(args):
    # get training and testing parameters
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # get data loaders
    test_dataloader, test_set = build_dataloader(cfg, args.image_id, training=False)
    cfg['MODEL']['out_dim'] = test_set.img_channels

    # get model
    device = torch.device("cuda:" + str(args.gpu) if (torch.cuda.is_available()) else "cpu")

    spec = importlib.util.spec_from_file_location('build_model_from_logdir', os.path.join(args.logdir, 'models', '__init__.py'))
    build_model_from_logdir = importlib.util.module_from_spec(spec)
    sys.modules['build_model_from_logdir'] = build_model_from_logdir
    spec.loader.exec_module(build_model_from_logdir)
    SINR, criterion = build_model_from_logdir.build_model_from_logdir(args.logdir, cfg, cfg['LOSS']).get()
    SINR.to(device)

    model_dir = os.path.join(args.logdir, 'trained_models')
    output_dir = os.path.join(args.logdir, 'vis')
    recon_img_outdir = os.path.join(args.logdir, 'reconstructed_images')
    os.makedirs(output_dir, exist_ok=True)
    # get loss


    _, test_data = next(enumerate(test_dataloader))
    SINR.eval()
    coords, gt_img, dino = test_data['coords'].to(device), test_data['gt_img'].to(device), test_data['dino'].to(device)

    epoch_n_eval = cfg['TESTING']['epoch_n_eval']
    eval_epochs = range(epoch_n_eval[0], epoch_n_eval[1], epoch_n_eval[2])

    psnr_module = PSNR()
    mse_module = RGBReconLossSingle()
    eval_metrics = {'Epochs': [], 'PSNR': [], 'MSE': []}

    for epoch in eval_epochs:
        print("Visualizing  {} epoch {}".format(args.image_id, epoch))

        model_filename = os.path.join(model_dir, '%s_model_%d.pth' % (cfg['MODEL']['model_name'], epoch))
        SINR.load_state_dict(torch.load(model_filename, map_location=device))
        SINR.to(device)

        coords.requires_grad_()
        output_pred = SINR(coords, dino=dino, img=gt_img)
        # loss_dict = criterion(output_pred=output_pred,  coords=coords, gt={'img': gt_img, 'segment': test_data['segments']})

        img_dict = {}
        if "moe" in cfg['MODEL']['model_name']:
            recon_img = output_pred['selected_nonmanifold_pnts_pred'].detach().cpu().numpy().reshape(test_set.sidelength[0],
                                                                                             test_set.sidelength[1], -1)

            recon_img_tensor = output_pred['selected_nonmanifold_pnts_pred'].squeeze(0)
            selected_expert_idx = output_pred['nonmnfld_selected_expert_idx']
            q = output_pred['nonmnfld_q']
            if cfg['TESTING'].get('plot_q_grad', True):
                q_grad = utils.experts_gradient(coords, q).norm(2, dim=-1)
                q_grad = q_grad.gather(dim=1, index=selected_expert_idx[None, None, :]).cpu().detach().numpy().reshape(
                    test_set.sidelength[0], test_set.sidelength[1])
            else:
                q_grad = None
            selected_expert_idx = selected_expert_idx.cpu().detach().numpy().reshape(test_set.sidelength[0], test_set.sidelength[1])
            q = q.squeeze().cpu().detach().numpy().reshape(-1, test_set.sidelength[0], test_set.sidelength[1])
            (img_dict['experts'], img_dict['experts_heatmap'], img_dict['q_grad'], img_dict['q_image_array_list'],
             img_dict['q_distributions'], img_dict['q_dist_array'], _) = (
                vis.plot_rgb_experts(selected_expert_idx, q, q_grad, example_idx=0, clim=(0.0, 0.5)))
            img_dict['reconstructed_img_per_experts'] = []
            for i in range(output_pred['nonmanifold_pnts_pred'].shape[1]):
                recon_img_e = output_pred['nonmanifold_pnts_pred'][:, i].detach().cpu().numpy().reshape(test_set.sidelength[0],
                                                                                             test_set.sidelength[1], -1)
                recon_img_e = convert_to_uint8(recon_img_e)
                img_dict['reconstructed_img_per_experts'].append(recon_img_e)
        else:
            recon_img = output_pred['nonmanifold_pnts_pred'].permute(0, 2, 1).detach().cpu().numpy().reshape(test_set.sidelength[0],
                                                                                    test_set.sidelength[1], -1)
            recon_img_tensor = output_pred['nonmanifold_pnts_pred']

        img_gradient = diff_operators.gradient(recon_img_tensor, coords)
        img_laplace = diff_operators.laplace(recon_img_tensor, coords)
        pred_grad = dataio.grads2img(dataio.lin2img(img_gradient,
                                                    image_resolution=test_set.sidelength)).permute(1, 2, 0).squeeze().detach().cpu().numpy()

        pred_lapl = cv2.cvtColor(cv2.applyColorMap(dataio.to_uint8(dataio.rescale_img(
            dataio.lin2img(img_laplace, image_resolution=test_set.sidelength), perc=2).permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()),
                                                   cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)

        img_dict['gradient_img'] = convert_to_uint8(pred_grad)
        img_dict['laplacian_img'] = pred_lapl

        error_img = (recon_img - gt_img.cpu().numpy().reshape(test_set.sidelength[0], test_set.sidelength[1], -1))**2

        img_dict['error_img'] = convert_to_uint8(error_img)
        img_dict['reconstructed_img'] = convert_to_uint8(recon_img).squeeze() #.transpose(1, 0, 2)

        eval_metrics['Epochs'].append(epoch)
        mse = mse_module.compute_loss(torch.tensor(recon_img, device=gt_img.device),
                                      gt_img.reshape(test_set.sidelength[0], test_set.sidelength[1], -1))
        psnr = psnr_module(gt_img.reshape(test_set.sidelength[0], test_set.sidelength[1], -1), mse)
        eval_metrics['MSE'].append(mse.item())
        eval_metrics['PSNR'].append(psnr.item())
        eval_metric_img_dict = vis.plot_eval_metrics(eval_metrics, data_range={'x':[0, max(eval_epochs)],
                                                                               'y':[0, 100]})
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

        del img_laplace, img_gradient # free memory

    # save the evaluation metrics to a file
    eval_metrics_filename = os.path.join(args.logdir, 'eval_metrics.pickle')
    with open(eval_metrics_filename, 'wb') as f:
        pickle.dump(eval_metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(eval_metrics)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Testing RGB MoE INR')
    parser.add_argument('--config', default='../configs/config_RGB.yaml', type=str, help='config file')
    parser.add_argument('--logdir', default='./log/debug_rgb', type=str, help='path to log firectory')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index to use')
    parser.add_argument('--image_id', default='kodim19.png', type=str, help='shape to load')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)


