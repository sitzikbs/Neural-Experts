# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os
import os.path
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import torch
import open3d as o3d
import yaml
from pykeops.torch import LazyTensor
import time
import pickle
import math

def sample_farthest_points_naive_new(points, K, selected_idx = None):
    P, D = points.shape
    device = points.device

    # Idxes of sampled points, initialised to -1, shape: (K,)
    sample_idx_batch = torch.full(
        (K,), fill_value=-1, dtype=torch.int64, device=device
    )

    # Choose the first point as the first chosen/sampled point
    if selected_idx is None:
        # selected_idx = 0
        selected_idx = points.norm(dim=-1).argmax().item()
    sample_idx_batch[0] = selected_idx

    # Closest dist of each point to all chosen points, initialised to inf, shape: (P,)
    closest_dists = torch.full(
        (P,), float("inf"), dtype=torch.float32, device=device
    )

    # Iteratively select points for 1...K-1
    for i in range(1, K):
        # Calc dist of last chosen point to all points
        dist_to_last_selected = ((points[selected_idx, :] - points) ** 2).sum(-1)  # (P - i)

        # Update dist of each point to all chosen points
        closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

        # Sample the point with maximum dist. Note chosen points have dist 0
        selected_idx = torch.argmax(closest_dists)
        sample_idx_batch[i] = selected_idx

    # Return (K, D) subsampled points and  (K,indices
    return points[sample_idx_batch], sample_idx_batch

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

class SDFdataset3D(data.Dataset):
    def __init__(self, cfg, shape_filename):

        self.dataset_path = cfg['DATA']['dataset_path']
        self.file_path = os.path.join(self.dataset_path, shape_filename)
        self.shape_filename = shape_filename
        self.n_points = cfg['TRAINING']['n_points']
        self.grid_res = cfg['TRAINING']['grid_res']
        self.grid_range = cfg['TRAINING']['grid_range']
        self.device = cfg['device']

        self.segmentation_mode = cfg['TRAINING']['segmentation_mode']

        # load data
        if '.ply' in self.file_path:
            self.o3d_point_cloud = o3d.io.read_point_cloud(self.file_path)
            # Returns points on the manifold (after centering and uniform scaling)
            points = np.asarray(self.o3d_point_cloud.points, dtype=np.float32)
            normals = np.asarray(self.o3d_point_cloud.normals, dtype=np.float32)
        elif '.xyz' in self.file_path:
            pointcloud = np.genfromtxt(self.file_path).astype(np.float32)
            points = pointcloud[:, :3]
            normals = pointcloud[:, 3:]
        else:
            raise NotImplementedError
        normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
        self.initial_points = points

        # center and scale point cloud. Sets self.cp and self.scale
        self.points = self.scale_points(self.initial_points)
        self.mnfld_n = normals

        self.shuffled_idxes = np.random.permutation(self.points.shape[0])
        self.shuffled_points = torch.tensor(self.points[self.shuffled_idxes]).to(self.device) # to allow pinning
        self.shuffled_normals = torch.tensor(self.mnfld_n[self.shuffled_idxes]).to(self.device) # to allow pinning
        
        self.setupFastKNN()

        
        self.grid_points = torch.tensor(self.get_grid_points(self.grid_range, self.grid_res)) # (grid_res**3, 3)
        if self.segmentation_mode:
            self.grid_shuffled_idxes = np.random.permutation(self.grid_points.shape[0])
            self.shuffled_grid_points = self.grid_points[self.grid_shuffled_idxes].to(self.device)
        
        # self.grid_sdfs, self.grid_normals = self.estimate_sdf_and_normals(self.grid_points, return_numpy=False)
        save_est_path = f'{shape_filename.split(".")[0]}_grid_estimates_{self.grid_res}.pkl'
        if os.path.exists(save_est_path):
            with open(save_est_path, 'rb') as fp:
                self.grid_sdfs, self.grid_normals = pickle.load(fp)
            print(f'Loaded grid estimates from {save_est_path}')
        else:
            t0 = time.time()
            self.grid_sdfs, self.grid_normals = self.estimate_sdf_and_normals(self.grid_points, return_numpy=False)
            print(f'Computing grid estimates took {time.time()-t0:.3f}s')
            with open(save_est_path, 'wb') as fp:
                pickle.dump((self.grid_sdfs, self.grid_normals), fp)

        # compute segmentation
        if cfg['DATA']['segmentation_type'] == 'random_balanced':
            self.n_segments = cfg['DATA']['n_segments']
            self.grid_patch_size = cfg['DATA']['grid_patch_size']
            assert np.sqrt(self.grid_patch_size) % 1 == 0.0, f'self.grid_patch_size={self.grid_patch_size} must be a square number'
            patch_side_length = int(np.sqrt(self.grid_patch_size))
            assert self.grid_res % patch_side_length == 0, f'grid_res={self.grid_res} must be divisible by patch size={patch_side_length}'
            self.segments = torch.tensor(self.random_balanced_segmentation()).to(self.device)
        else:
            raise NotImplementedError

    def scale_points(self, points):
        # points: (n_points, d)
        assert len(points.shape) == 2, points.shape

        if not hasattr(self, 'cp'):
            print('setting cp')
            self.cp = points.mean(axis=0, keepdims=True)
        scaled_points = points - self.cp

        if not hasattr(self, 'scale'):
            print('setting scale')
            # self.scale = np.linalg.norm(scaled_points, axis=-1).max(-1) # alternative scaling option
            self.scale = np.abs(scaled_points).max()
        scaled_points /= self.scale

        return scaled_points
    
    def unscale_points(self, points):
        # points: (n_points, d)
        assert len(points.shape) == 2, points.shape

        return points*self.scale + self.cp

    def random_balanced_segmentation(self):
        ''' Create a random segmentation of rows and columns.'''
        # full array is (grid_res, grid_res), but we assign segments in patches of size patch_side_length
        # thus we assign patches on a (grid_res/patch_side_length, grid_res/patch_side_length) grid and then
        # upsize
        patch_side_length = int(np.sqrt(self.grid_patch_size))
        reduced_grid_res = self.grid_res // patch_side_length
        segments = np.random.randint(0, self.n_segments, (reduced_grid_res, reduced_grid_res, reduced_grid_res))
        segments = segments.repeat(patch_side_length, axis=2).repeat(patch_side_length, axis=1).repeat(patch_side_length, axis=0)
        return segments.flatten()[None, :]
    
    def query_segmentation(self, query, return_numpy=True):
        # query: (n_q, 3)
        assert len(query.shape) == 2, query.shape
        grid_res = int(round((self.segments.shape[-1]) ** (1. / 3)))
        idxes = query.clip(-1*self.grid_range + 1e-5, self.grid_range - 1e-5)
        idxes = idxes + self.grid_range  # now in range [0, 2*self.grid_range]
        idxes = idxes * grid_res / (2*self.grid_range) # now in range [0,grid_res]
        idxes = torch.floor(idxes).long() # now ints in 0,...,grid_res-1
        idxes = idxes[:,0]*(grid_res**2) + idxes[:,1]*grid_res + idxes[:,2]
        return self.segments[0,idxes] # (n_q,)
    
    def get_grid_points(self, grid_range, grid_res):
        # generate grid points
        x, y, z = np.linspace(-grid_range, grid_range, grid_res).astype(np.float32), \
            np.linspace(-grid_range, grid_range, grid_res).astype(np.float32), \
            np.linspace(-grid_range, grid_range, grid_res).astype(np.float32)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        xx, yy, zz = xx.ravel(), yy.ravel(), zz.ravel()
        return np.stack([xx, yy, zz], axis=1).astype('f') # (grid_res**3, 3)
    
    def setupFastKNN(self):
        self.num_fps = 10000
        self.num_neigh = 100

        save_est_path = f'{self.shape_filename.split(".")[0]}_fps.pkl'
        if os.path.exists(save_est_path):
            with open(save_est_path, 'rb') as fp:
                self.fps_points, self.fps_idxes = pickle.load(fp)
            print(f'Loaded FPS from {save_est_path}')
        else:
            t0 = time.time()
            # (n_fps, )
            self.fps_points, self.fps_idxes = sample_farthest_points_naive_new(self.shuffled_points, self.num_fps)
            print(f'FPS took {time.time()-t0:.3f}s')
            with open(save_est_path, 'wb') as fp:
                pickle.dump((self.fps_points, self.fps_idxes), fp)

        save_est_path = f'{self.shape_filename.split(".")[0]}_m2neigh_{self.num_fps}_{self.num_neigh}.pkl'
        if os.path.exists(save_est_path):
            with open(save_est_path, 'rb') as fp:
                self.idxes_m2neigh = pickle.load(fp)
            print(f'Loaded PC neighbours from {save_est_path}')
        else:
            t0 = time.time()
            # (n_m, n_neigh)
            _, self.idxes_m2neigh = keops_knn(self.shuffled_points, self.shuffled_points, k=self.num_neigh, return_numpy=False)
            print(f'Generating PC neighbours {tuple(self.idxes_m2neigh.shape)} took {time.time()-t0:.3f}s')
            with open(save_est_path, 'wb') as fp:
                pickle.dump(self.idxes_m2neigh, fp)

        self.idxes_fps2neigh = self.idxes_m2neigh[self.fps_idxes]
    
    def refine_query_nn(self, query, idxes_q2pot_nn):
        # idxes_q2pot_nn has a list of idxes (into self.shuffled_points) of potential closest points to each query point
        # query is (n_q, 3), idxes_q2pot_nn is (n_m, p)
        potential_nn = self.shuffled_points[idxes_q2pot_nn] # (n_q, n_neigh, 3)
        D_ij = ((query[:, None, :] - potential_nn)**2).sum(dim=-1).sqrt() # (n_q, n_neigh)
        # find closest index into potential_nn
        closest_idxes = D_ij.argmin(dim=-1) # (n_q)
        # convert to index into self.shuffled_points
        closest_idxes = torch.take_along_dim(idxes_q2pot_nn, closest_idxes[:,None], dim=1).squeeze() # (n_q, )
        return closest_idxes
    
    def get_pcd_nn(self, query):
        # query is (n_q, 3)
        n_q = query.shape[0]
        if not torch.is_tensor(query):
            query = torch.tensor(query)
        query = query.to(self.device)

        # find which fps points are relevant, (n_q,) in range(0,n_fps)
        # idxes_q2fps is (n_q, k=10)
        dists, idxes_q2fps = keops_knn(query, self.fps_points, k=10, return_numpy=False)

        # (n_q, k*n_neigh)
        idxes_q2pot_nn = self.idxes_fps2neigh[idxes_q2fps].reshape(n_q, -1)
        # (n_q,)
        nn_idxes = self.refine_query_nn(query, idxes_q2pot_nn)
        for it_num in range(3):
            idxes_q2pot_nn = self.idxes_m2neigh[nn_idxes]
            new_nn_idxes = self.refine_query_nn(query, idxes_q2pot_nn)
            if (nn_idxes == new_nn_idxes).all():
                break
            else:
                nn_idxes = new_nn_idxes
        # print(f'took {it_num} iters for pcd_nn to converge')
        nn_dists = (query - self.shuffled_points[nn_idxes]).norm(dim=-1) # (n_q,)
        return nn_dists, nn_idxes

    def estimate_sdf_and_normals(self, query, return_numpy=True):
        # t0 = time.time()
        # query: (n_q, 2)
        assert len(query.shape) == 2, query.shape
        if not torch.is_tensor(query):
            query = torch.tensor(query)
        query = query.to(self.device)

        mnfld_points = self.shuffled_points # (n, d)
        mnfld_normals = self.shuffled_normals # (n, d)

        k = 1
        # dists_a2b, idxes_a2b = keops_knn(query, mnfld_points, k=k, return_numpy=False)
        chunksize = 100000
        num_q = query.shape[0]
        if num_q <= chunksize:
            dists_a2b, idxes_a2b = self.get_pcd_nn(query)
        else:
            dists_a2b = []
            idxes_a2b = []
            for i in range(math.ceil(num_q/chunksize)):
                dists_a2b_i, idxes_a2b_i = self.get_pcd_nn(query[i*chunksize : (i+1)*chunksize])
                dists_a2b.append(dists_a2b_i)
                idxes_a2b.append(idxes_a2b_i)
            dists_a2b = torch.cat(dists_a2b, dim=0)
            idxes_a2b = torch.cat(idxes_a2b, dim=0)

        # if k>1 then (n_a, k), (n_a, k) else (n_a,), (n_a, )
        if k == 1:
            est_dists = dists_a2b # (n_a,)
            mean_normals = mnfld_normals[idxes_a2b]
            disps = query - mnfld_points[idxes_a2b]
        else:   
            est_dists = dists_a2b[:,0]
            mean_normals = mnfld_normals[idxes_a2b].mean(axis=1)
            # max_idxes = idxes_a2b[:,0]
            # disps = query - mnfld_points[max_idxes]
            disps = query - mnfld_points[idxes_a2b].mean(1)
        est_signs = torch.sign(torch.einsum('ij,ij->i', disps, mean_normals))
        # import pdb; pdb.set_trace()
        # est_signs = torch.sign((disps * mean_normals).sum(dim=-1))
        # est_normals = (query - mnfld_points[max_idxes]) / est_dists[:, None] * est_signs[:, None]
        # est_normals[est_dists < 1e-3] = mean_normals[est_dists < 1e-3]
        est_normals = mean_normals
        est_sdfs = est_dists * est_signs
        # est_sdfs = (disps * mean_normals).sum(dim=-1) + np.random.randn(est_sdfs.shape[0])
        # print(f'estimating sdf took {time.time()-t0:.3f}s')
        if return_numpy:
            return est_sdfs.numpy(), est_normals.numpy()
        return est_sdfs, est_normals

    def __getitem__(self, index):
        if self.segmentation_mode:
            # use grid samples
            num_bs = self.shuffled_grid_points.shape[0] // self.n_points
            b_i = index % num_bs
            grid_samples = self.shuffled_grid_points[b_i*self.n_points : (b_i+1)*self.n_points].clone()
            grid_samples += torch.tensor(np.random.laplace(scale=2e-3, size=tuple(grid_samples.shape)).astype(np.float32)).to(self.device)
            uniform_samples = (torch.rand_like(grid_samples) - 0.5) * 1.1
            nonmnfld_points = torch.cat([grid_samples, uniform_samples], dim=0) # (2*self.n_points, 2)
            nonmnfld_segments = self.query_segmentation(nonmnfld_points, return_numpy=False)
            mnfld_points = nonmnfld_points[:3]
            return {
                'mnfld_points': mnfld_points, # (n_points, d)
                "nonmnfld_points": nonmnfld_points,  # (n_nm, d)
                'nonmnfld_segments_gt': nonmnfld_segments,  # (n_nm, )
            }
        else:
            num_bs = self.shuffled_points.shape[0] // self.n_points
            b_i = index % num_bs
            # b_i = 0
            mnfld_points = self.shuffled_points[b_i*self.n_points : (b_i+1)*self.n_points].clone()
            # mnfld_n = self.shuffled_normals[b_i*self.n_points : (b_i+1)*self.n_points]
            # mnfld_segments_gt = self.shuffled_segments[b_i*self.n_points : (b_i+1)*self.n_points]
            
            b_i = (index+1) % num_bs
            fine_points = self.shuffled_points[b_i*self.n_points : (b_i+1)*self.n_points].clone()
            b_i = (index+2) % num_bs
            course_points = self.shuffled_points[b_i*self.n_points : (b_i+1)*self.n_points].clone()

            fine_points += torch.tensor(np.random.laplace(scale=2e-3, size=(self.n_points, mnfld_points.shape[-1])).astype(np.float32)).to(self.device)
            course_points += torch.tensor(np.random.laplace(scale=2e-1, size=(self.n_points, mnfld_points.shape[-1])).astype(np.float32)).to(self.device)

            nonmnfld_points = torch.cat([fine_points, course_points], dim=0) # (2*self.n_points, 2)
            # nonmnfld_points = fine_points

            # wrap around any points that are sampled out of bounds
            nonmnfld_points[nonmnfld_points > self.grid_range] = torch.remainder(nonmnfld_points[nonmnfld_points > self.grid_range], self.grid_range)
            nonmnfld_points[nonmnfld_points < -1*self.grid_range] = torch.remainder(nonmnfld_points[nonmnfld_points < -1*self.grid_range], -1*self.grid_range)

            nonmnfld_sdf, nonmnfld_n = self.estimate_sdf_and_normals(nonmnfld_points, return_numpy=False)
            
            return {
                'mnfld_points': mnfld_points, # (n_points, d)
                "nonmnfld_points": nonmnfld_points,  # (n_nm, d)
                "nonmnfld_sdf": nonmnfld_sdf,  # (n_nm, )
            }

    def __len__(self):
        return np.iinfo(np.int32).max
