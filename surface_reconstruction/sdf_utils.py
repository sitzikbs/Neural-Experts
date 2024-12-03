import os
import sys
import time
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np

from pykeops.torch import LazyTensor

import trimesh
import open3d as o3d
from skimage import measure

import utils.visualizations as vis
import utils.utils as utils
from PIL import Image

# @profile
def keops_knn(pts_a, pts_b, k, force_cuda=False, return_numpy=False):
    if not torch.is_tensor(pts_a):
        pts_a = torch.tensor(pts_a)
    if not torch.is_tensor(pts_b):
        pts_b = torch.tensor(pts_b)
    
    if force_cuda:
        pts_a_i = LazyTensor(pts_a.cuda()[:, None, :])  # (M, 1, D) LazyTensor
        pts_b_j = LazyTensor(pts_b.cuda()[None, :, :])  # (1, N, D) LazyTensor
    else:
        pts_a_i = LazyTensor(pts_a[:, None, :])  # (M, 1, D) LazyTensor
        pts_b_j = LazyTensor(pts_b[None, :, :])  # (1, N, D) LazyTensor

    D_ij = ((pts_a_i - pts_b_j) ** 2).sum(dim=-1).sqrt() # (M, N)
    dists_a2b, idxes_a2b = D_ij.Kmin_argKmin(k, dim=1) # (M, k), (M, k)
    dists_a2b = dists_a2b.squeeze()
    idxes_a2b = idxes_a2b.squeeze()
    if return_numpy:
        return dists_a2b.numpy(), idxes_a2b.numpy()
    else:
        return dists_a2b, idxes_a2b

def dict2device(in_dict, device, *args, **kwargs):
    out_dict = {}
    for key, value in in_dict.items():
        if isinstance(value, torch.Tensor):
            out_dict.update({key: value.to(device, *args, **kwargs)})
        elif isinstance(value, dict):
            out_dict.update({key: dict2device(value, device, *args, **kwargs)})
        elif isinstance(value, list) or isinstance(value, tuple):
            if isinstance(value[0], torch.Tensor):
                out_dict.update({key: [v.to(device, *args, **kwargs) for v in value]})
            else:
                out_dict.update({key: value})
        else:
            out_dict.update({key: value})
    return out_dict

# @profile
def compute_full_grid(grid_points_flattened, device, SINR, process_size=256*256*2):
    num_grid_points = grid_points_flattened.shape[0]
    grid_points_flattened = grid_points_flattened.to(device)

    grid_sdfs = []
    with torch.no_grad():
        for i in range(math.ceil((num_grid_points / process_size))):
            current_batch = grid_points_flattened[i*process_size:(i+1)*process_size]
            output_pred_grid = SINR(current_batch[None, :])
            if 'selected_nonmanifold_pnts_pred' in output_pred_grid:
                grid_pred = output_pred_grid['selected_nonmanifold_pnts_pred'].reshape(-1)
            else:
                grid_pred = output_pred_grid['nonmanifold_pnts_pred'].reshape(-1)
                
            grid_sdfs.append(grid_pred)
    grid_sdfs = torch.cat(grid_sdfs)
    return {'nonmanifold_pnts_pred': grid_sdfs}

def pred_sdf_to_mesh(pred_sdf, output_ply_filepath, grid_res, grid_points, dataset, title):
    # print("pred_sdf_to_mesh started")
    t0 = time.time()
    # grid_res = dataset.grid_res
    cp = dataset.cp
    scale = dataset.scale
    # grid_points = dataset.grid_points.numpy() # (gs**3, 3)
    grid_points = grid_points.numpy()
    # input is (gs*gs*gs,1)
    # print('Creating Mesh')
    assert pred_sdf.shape == (grid_res**3, ), pred_sdf.shape
    if torch.is_tensor(pred_sdf):
        pred_sdf = pred_sdf.detach().cpu().numpy()
    pred_sdf = pred_sdf.reshape(grid_res,grid_res,grid_res) # (gs,gs,gs)
    spacing = abs(grid_points[0][-1] - grid_points[1][-1])
    
    # (nv, 3), (nf, 3), (nv, 3), (nv,)
    verts, faces, normals, values = measure.marching_cubes(volume=pred_sdf, level=0.0,
                                                spacing=(spacing, spacing, spacing))
    verts = verts + grid_points[0].reshape(1,3) # (n_v,3)
    verts = verts * scale + cp
    
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=values, validate=True)
    vis.plot_mesh(None, mesh=mesh, output_ply_path=output_ply_filepath, show_ax=False,
            title_txt=title, show=False)
    # print("pred_sdf_to_mesh ended, ({:.3f})s".format(time.time()-t0))
    return mesh


def pred_sdf_to_coloured_mesh(pred_sdf, output_ply_filepath, grid_res, grid_points, dataset, title, device, SINR, process_size=256*256*2):
    # print("pred_sdf_to_mesh started")
    t0 = time.time()
    # grid_res = dataset.grid_res
    cp = dataset.cp
    scale = dataset.scale
    # grid_points = dataset.grid_points.numpy() # (gs**3, 3)
    grid_points = grid_points.numpy()
    # input is (gs*gs*gs,1)
    # print('Creating Mesh')
    assert pred_sdf.shape == (grid_res**3, ), pred_sdf.shape
    if torch.is_tensor(pred_sdf):
        pred_sdf = pred_sdf.detach().cpu().numpy()
    pred_sdf = pred_sdf.reshape(grid_res,grid_res,grid_res) # (gs,gs,gs)
    spacing = abs(grid_points[0][-1] - grid_points[1][-1])
    
    # (nv, 3), (nf, 3), (nv, 3), (nv,)
    verts, faces, normals, values = measure.marching_cubes(volume=pred_sdf, level=0.0,
                                                spacing=(spacing, spacing, spacing))
    verts = verts + grid_points[0].reshape(1,3) # (n_v,3)

    verts_cuda = torch.tensor(verts, dtype=torch.float32).to(device)
    num_verts = verts_cuda.shape[0]
    vert_expert_idx = []
    with torch.no_grad():
        for i in range(math.ceil((num_verts / process_size))):
            current_batch = verts_cuda[i*process_size:(i+1)*process_size]
            output_pred_verts = SINR(current_batch[None, :])
            expert_idx = output_pred_verts['nonmnfld_selected_expert_idx']                
            vert_expert_idx.append(expert_idx)
    vert_expert_idx = torch.cat(vert_expert_idx).cpu().numpy()

    verts = verts * scale + cp
    
    # mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=values, validate=True)
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=values)
    vis.plot_mesh(None, mesh=mesh, output_ply_path=output_ply_filepath, show_ax=False, title_txt=title, show=False)
    # now do colours based on expert indices
    mesh.visual.vertex_colors = trimesh.visual.interpolate(vert_expert_idx, color_map='Set1')
    coloured_filepath = output_ply_filepath.replace('.ply', '_coloured.ply')
    vis.plot_mesh(None, mesh=mesh, output_ply_path=coloured_filepath, show_ax=False, title_txt=title, show=False)

    # print("pred_sdf_to_mesh ended, ({:.3f})s".format(time.time()-t0))
    return mesh

def grid_pred2metrics(grid_pred, grid_sdfs_gt, grid_res, dataset, device, metrics_dict={}):
    # grid_sdfs_gt = dataset.grid_sdfs.to(device)
    # grid_normals_gt = dataset.grid_normals.to(device)
    grid_sdfs_gt = grid_sdfs_gt.to(device)
    if 'selected_nonmanifold_pnts_pred' in grid_pred:
        grid_sdfs_pred = grid_pred['selected_nonmanifold_pnts_pred'].squeeze()
    else:
        grid_sdfs_pred = grid_pred['nonmanifold_pnts_pred'].squeeze()
    diff = (grid_sdfs_gt - grid_sdfs_pred).abs()
    sdf_mse = (diff ** 2).mean()
    grid_occ_gt = grid_sdfs_gt < 0
    grid_occ_pred = grid_sdfs_pred < 0
    iou = (grid_occ_gt & grid_occ_pred).sum() / (grid_occ_gt | grid_occ_pred).sum()
    metrics_dict[f'IoU_{grid_res}'] = iou
    metrics_dict[f'sdf_mse_{grid_res}'] = sdf_mse


    narrow_idxes = grid_sdfs_gt.abs() < 0.1
    grid_occ_gt_narrow = grid_occ_gt[narrow_idxes]
    grid_occ_pred_narrow = grid_occ_pred[narrow_idxes]
    iou_narrow = (grid_occ_gt_narrow & grid_occ_pred_narrow).sum() / (grid_occ_gt_narrow | grid_occ_pred_narrow).sum()
    metrics_dict[f'IoU_<0.1'] = iou_narrow
    narrow_idxes = grid_sdfs_gt.abs() < 0.01
    grid_occ_gt_narrow = grid_occ_gt[narrow_idxes]
    grid_occ_pred_narrow = grid_occ_pred[narrow_idxes]
    iou_narrow = (grid_occ_gt_narrow & grid_occ_pred_narrow).sum() / (grid_occ_gt_narrow | grid_occ_pred_narrow).sum()
    metrics_dict[f'IoU_<0.01'] = iou_narrow
    narrow_idxes = grid_sdfs_gt.abs() < 0.005
    grid_occ_gt_narrow = grid_occ_gt[narrow_idxes]
    grid_occ_pred_narrow = grid_occ_pred[narrow_idxes]
    iou_narrow = (grid_occ_gt_narrow & grid_occ_pred_narrow).sum() / (grid_occ_gt_narrow | grid_occ_pred_narrow).sum()
    metrics_dict[f'IoU_<0.005'] = iou_narrow
    narrow_idxes = grid_sdfs_gt.abs() < 0.001
    grid_occ_gt_narrow = grid_occ_gt[narrow_idxes]
    grid_occ_pred_narrow = grid_occ_pred[narrow_idxes]
    iou_narrow = (grid_occ_gt_narrow & grid_occ_pred_narrow).sum() / (grid_occ_gt_narrow | grid_occ_pred_narrow).sum()
    metrics_dict[f'IoU_<0.001'] = iou_narrow
    diff_idxes = (grid_occ_gt != grid_occ_pred)
    metrics_dict[f'furthest_sign_mistake'] = grid_sdfs_gt[diff_idxes].abs().max()
    # print(metrics_dict)
    # import pdb; pdb.set_trace()

    # metrics_dict['diff_max'] = diff.max()
    # metrics_dict['diff_mean'] = diff.mean()
    # metrics_dict['diff_median'] = diff.median()
    # metrics_dict['diff_min'] = diff.min()
    # # import pdb; pdb.set_trace()
    # # diff>0.5
    # # sdf_large = grid_sdfs_gt[diff>0.5]
    # # sdf_large = grid_sdfs_gt[diff>0.5]
    # sdf_large = grid_sdfs_gt[grid_occ_gt != grid_occ_pred]
    # metrics_dict['sdf_large_max'] = sdf_large.max()
    # metrics_dict['sdf_large_mean'] = sdf_large.mean()
    # metrics_dict['sdf_large_median'] = sdf_large.median()
    # metrics_dict['sdf_large_min'] = sdf_large.min()
    # metrics_dict['oc_diff'] = (grid_occ_gt != grid_occ_pred).sum()
    return metrics_dict

def mesh2metrics(mesh, dataset, device, metrics_dict={}):
    # import pdb; pdb.set_trace()
    n_points = dataset.initial_points.shape[0]
    gt_samples = dataset.initial_points[np.random.permutation(n_points)[:300000]]
    gt_samples = gt_samples.astype(np.float32)
    pred_samples, _ = trimesh.sample.sample_surface(mesh, 300000)
    pred_samples = pred_samples.astype(np.float32)

    gt_samples = dataset.scale_points(gt_samples)
    pred_samples = dataset.scale_points(pred_samples)
    
    dists_a2b, _ = keops_knn(pred_samples, gt_samples, 1)
    dists_b2a, _ = keops_knn(gt_samples, pred_samples, 1)
    sq_chamfer_dist = (dists_a2b**2).mean() + (dists_b2a**2).mean()
    hausdorff_dist = dists_a2b.max() + dists_b2a.max()
    metrics_dict['SqChamfer'] = sq_chamfer_dist
    metrics_dict['Hausdorrf'] = hausdorff_dist
    return metrics_dict



