# Code adapted from https://github.com/vsitzmann/siren/blob/master/dataio.py#L818
# This is a dataset class for RGB images. It is used to load the RGB images and convert them to a format that can be used by the model.
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import os
import torch.nn as nn
import torchvision
import cv2

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


class RGBINRDataset(torch.utils.data.Dataset):
    def __init__(self, data_cfg, img_id):
        self.downsample_factor = data_cfg['downsample_factor']
        self.copy_to_gpu = data_cfg['copy_to_gpu']
        self.img = Image.open(os.path.join(data_cfg['dataset_path'], img_id))
        self.sidelength = (self.img.width, self.img.height)
        self.bandlimit = data_cfg.get('bandlimit', None)

        if self.downsample_factor > 1:
            self.sidelength = (int(self.sidelength[0] / self.downsample_factor),
                               int(self.sidelength[1] / self.downsample_factor))
            self.img = self.img.resize(self.sidelength, resample=Image.LANCZOS)

        if not self.sidelength[0] == self.sidelength[1]:
            # crop the img around the center
            min_side = min(self.sidelength)
            self.img = self.img.crop((int((self.sidelength[0] - min_side) / 2),
                                      int((self.sidelength[1] - min_side) / 2),
                                      int((self.sidelength[0] + min_side) / 2),
                                      int((self.sidelength[1] + min_side) / 2)))
            self.sidelength = (min_side, min_side)

        if self.bandlimit is not None:
            # filter the image with a Gaussian blur
            self.img = self.apply_bandlimit(self.img)
            self.freq_domain = self.get_freq_domain()

        self.img = np.array(self.img)
        if len(self.img.shape) == 2:
            self.img_channels = 1
        else:
            self.img_channels = self.img.shape[-1]

        if self.copy_to_gpu:
            self.transform = Compose([
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
            ])
        else:
            self.transform = Compose([
                ToTensor(),
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
            ])

        self.mgrid = get_mgrid(self.sidelength)

        # compute segmentation
        if data_cfg['segmentation_type'] == 'kmeans':
            self.segments = self.get_kmeans_segments(data_cfg['n_segments'], data_cfg['kmeansspace'])
        elif data_cfg['segmentation_type'] == 'grid':
            self.n_segments = data_cfg['n_segments']
            self.grid_patch_size = data_cfg['grid_patch_size']
            patch_side_length = int(np.sqrt(self.grid_patch_size))
            assert self.sidelength[0] % patch_side_length == 0 and self.sidelength[1] % patch_side_length == 0, 'sidelength must be divisible by patch size'
            assert np.log2(self.grid_patch_size) % 1 == 0 or self.grid_patch_size == 1, 'patch size must be power of 2 or the value 1'
            self.segments = self.get_grid_segmentation()
        elif data_cfg['segmentation_type'] == 'raster':
            self.n_segments = data_cfg['n_segments']
            self.segments = self.raster_segmentation()
        elif data_cfg['segmentation_type'] == 'random_balanced':
            self.n_segments = data_cfg['n_segments']
            self.segments = self.random_balanced_segmentation()
        elif data_cfg['segmentation_type'] == 'circular':
            self.n_segments = data_cfg['n_segments']
            self.segments = self.circular_segmentation()
        elif data_cfg['segmentation_type'] == 'sam':
            self.n_segments = data_cfg['n_segments']
            self.sam_model_type = data_cfg['sam_model_type']
            self.sam_checkpoint = data_cfg['sam_checkpoint']
            self.segments = self.get_sam_segments()
        else:
            self.n_segments = 0
            self.segments = None


        self.get_dino = data_cfg['get_dino']
        if self.get_dino:
            self.dino = self.extract_dino_features() #deprecated
            self.dino_dim = self.dino.shape[-1]
        else:
            self.dino = None
            self.dino_dim = 0

        # copy the data to the gpu
        if self.copy_to_gpu:
            self.img = torch.tensor(self.img, dtype=torch.float32).to('cuda')
            self.mgrid = self.mgrid.to('cuda')
            if self.segments is not None:
                self.segments = torch.tensor(self.segments).to('cuda')
            if self.dino is not None:
                self.dino = self.dino.to('cuda')

    def apply_bandlimit(self, img):
        img = np.array(img)

        freq = int(self.bandlimit.split('_')[-1])
        if 'lowpass' in self.bandlimit:
            img_back = self. apply_gaussian_filter(img, freq, high_pass=False)
        elif 'highpass' in self.bandlimit:
            img_back = self. apply_gaussian_filter(img, freq, high_pass=True)

        return img_back

    def apply_filter_to_channel(self, channel, threshold_freq):
        # Convert the channel to float32
        channel_float32 = np.float32(channel)

        # Compute the Fourier Transform
        dft = cv2.dft(channel_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Create a Gaussian filter for low-pass or Laplacian filter for high-pass
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols, 2), np.float32)
        for i in range(rows):
            for j in range(cols):
                dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                mask[i, j] = np.exp(-((dist ** 2) / (2 * threshold_freq ** 2)))

        # Apply the filter in the frequency domain
        dft_shift_filtered = dft_shift * mask

        # Convert the result back to the spatial domain
        dft_filtered = np.fft.ifftshift(dft_shift_filtered)
        filtered_channel_float32 = cv2.idft(dft_filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # Convert the filtered channel back to uint8
        filtered_channel_uint8 = np.uint8(filtered_channel_float32)

        return filtered_channel_uint8

    def apply_gaussian_filter(self, image, threshold_freq, high_pass=False):
        # Split the image into color channels
        b, g, r = cv2.split(image)

        # Apply the Gaussian filter to each color channel
        b_filtered = self.apply_filter_to_channel(b, threshold_freq)
        g_filtered = self.apply_filter_to_channel(g, threshold_freq)
        r_filtered = self.apply_filter_to_channel(r, threshold_freq)

        # Merge the filtered color channels back into a color image
        filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))

        if high_pass:
            filtered_image = cv2.addWeighted(image, 1.5, filtered_image, -0.5, 0)
        return filtered_image

    def get_freq_domain(self):
        '''Computes the frequency domain of the image.'''
        # Compute the Fourier Transform of the images
        dft = cv2.dft(np.float32(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)),
                      flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        return magnitude_spectrum


    def resize(self, image, size=None, scale_factor=None):
        return nn.functional.interpolate(
            image,
            size=size,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )

    def extract_dino_features(self, device='cuda'):
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14") #TODO add support for other models: B,L,G
        feature_dim = model.embed_dim
        patch_size = 14

        for param in model.parameters():
            param.requires_grad = False

        x = torch.tensor(self.img).float().to(device)
        if len(x.shape) == 2:
            x = x.unsqueeze(2).repeat(1, 1, 3)
        x = x.unsqueeze(0).permute(0, 3, 1, 2)  # (B, C, H, W)
        x = torchvision.transforms.functional.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # make the image divisible by the patch size
        w, h = x.shape[2], x.shape[3]
        new_w, new_h = w - w % patch_size, h - h % patch_size
        x = x[:, :, :new_w, :new_h]  # resize to match required model input dims

        model.to(device)

        # Output will be (B, H * W, C)
        features = model.forward_features(x)["x_norm_patchtokens"]
        features = features.permute(0, 2, 1)
        features = features.reshape(-1, feature_dim, h // patch_size, w // patch_size) # (B, C, H, W)
        features = self.resize(features, size=(w, h))
        features = features.permute(0, 2, 3, 1)
        return features.detach().cpu().numpy()

    def reindex_segments(self, index_map):
        if  self.segments is not None :
            # reindex the segments to match the centroids
            temp_segments = self.segments.copy()
            for i, val in enumerate(index_map):
                self.segments[temp_segments == i] = val

    def get_kmeans_segments(self, num_clusters=4, kmeansspace='rgb'):
        from sklearn.cluster import KMeans

        if kmeansspace == 'rgb':
            kmeans_space = self.img.reshape(-1, self.img_channels)/255
        elif kmeansspace == 'rgbxy':
            kmeans_space = np.concatenate([0.25*self.mgrid, self.img.reshape(-1, self.img_channels)/255], axis=-1)
        else:
            raise NotImplementedError('kmeans space not implemented')
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(kmeans_space)

        return kmeans.labels_[None, :]

    def get_sam_segments(self):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        from sklearn.cluster import KMeans
        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
        sam.to(device="cuda")

        # mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=100) #TODO fix opencv qt bug so this will work
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(self.img)

        # sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        # segments = np.zeros([self.img.shape[0], self.img.shape[1],1], dtype=np.uint8)
        # # for i, mask in enumerate(sorted_masks):
        # #     segments[mask["segmentation"]] = i
        # for i in range(self.n_segments - 1):
        #     #expert 0 gets all the "leftover" segments
        #     segments[sorted_masks[i]["segmentation"]] = i + 1

        # Flatten the masks and apply K-means clustering
        flattened_masks = [mask["segmentation"].flatten() for mask in masks]
        kmeans = KMeans(n_clusters=self.n_segments).fit(flattened_masks)

        # Assuming `segments` is your 1D array of labels and `img` is your image
        segments = np.zeros(self.img.shape[0] * self.img.shape[1] , dtype=np.uint8)
        for i, mask in enumerate(flattened_masks):
            segments[mask] = kmeans.labels_[i]
        return segments.flatten()[None, :]


    def get_sanitycheck_segments(self):
        # create a simple grid pattern
        segments = np.zeros([1,self.sidelength[0]*self.sidelength[1]])
        return segments

    def get_grid_segmentation(self):
        '''Creates a grid segmentation of the image. The number of segments is determined by the n_segments parameter.'''
        segment_step = int(np.sqrt(self.n_segments))
        patch_side_length = int(np.sqrt(self.grid_patch_size))
        n_patches_per_side = (self.sidelength[0] // patch_side_length, self.sidelength[1] // patch_side_length)

        arr = np.arange(self.n_segments).reshape(segment_step, segment_step)
        tile_shape = arr.shape

        if n_patches_per_side[0] == 1 or n_patches_per_side[1] == 1:
            arr = arr.repeat(patch_side_length / segment_step, axis=1).repeat(patch_side_length / segment_step, axis=0)
            segments = arr
        else:
            arr = arr.repeat(patch_side_length, axis=1).repeat(patch_side_length, axis=0)
            segments = np.tile(arr, (int(n_patches_per_side[1] / tile_shape[1]), int( n_patches_per_side[0] / tile_shape[0])))
        return segments.flatten()[None, :]

    def raster_segmentation(self):
        ''' Create a segmentation of rows and columns.'''
        segment_step = self.n_segments
        assert self.sidelength[0] % segment_step == 0 and self.sidelength[1] % segment_step == 0, 'sidelength must be divisible by the number of segments'
        segments = np.tile(np.arange(segment_step)[:, None].repeat(self.sidelength[1], 1),
                           [int(self.sidelength[0] / segment_step), 1])
        return segments[None, :].flatten()[None, :]

    def random_balanced_segmentation(self):
        ''' Create a random segmentation of rows and columns.'''
        segments = np.random.randint(0, self.n_segments, (self.sidelength[0], self.sidelength[1]))
        return segments[None, :].flatten()[None, :]

    def circular_segmentation(self):
        ''' Create a circular segmentation around the image center.'''

        # Get the center of the array
        center = np.array(self.sidelength) // 2
        indices = np.indices(self.sidelength).transpose(1, 2, 0)
        distances = np.linalg.norm(indices - center, axis=-1)

        segments = np.zeros_like(distances).astype(int)
        for r in range(int(np.sqrt(2*max(self.sidelength)**2)//2)):
            val = int(r % self.n_segments)
            # Start at point (r, 0)
            x, y = r, 0
            # Loop until we reach an eighth of the circle
            while x >= y:
                # Calculate mirrored points in the other seven octants
                points = [(x, y), (-x, y), (x, -y), (-x, -y), (y, x), (-y, x), (y, -x), (-y, -x)]
                # Set the value at all these points in the array
                for point in points:
                    new_point = (center[0] + point[0], center[1] + point[1])
                    # Check if the point is within the bounds of the array
                    if (0 <= new_point[0] < segments.shape[0]) and (0 <= new_point[1] < segments.shape[1]):
                        segments[new_point] = val
                # Take a step upwards or diagonally
                if x * x + (y + 1) * (y + 1) <= r * r:
                    y += 1
                else:
                    x -= 1
        return segments.flatten()[None, :]

    def __len__(self):
        return 1 # single image

    def __getitem__(self, idx):
        img = self.transform(self.img)
        img = img.permute(1, 2, 0).reshape(-1, self.img_channels) # flatten the image to match coordinates
        if self.segments is None:
            segments = torch.tensor(0)
        else:
            segments = self.segments

        if self.get_dino:
            dino = self.dino.reshape(-1, self.dino.shape[-1])
        else:
            dino = 0

        out_dict = {'coords': self.mgrid, 'gt_img': img, 'segments': segments, 'dino': dino}
        return out_dict


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import yaml
    dataset_path = '../data'
    img_id ='kodim19.png'
    cfg_path = '/home/sitzikbs/PycharmProjects/DiGS_MOE/DiGS/configs/config_RGB.yaml'
    cfg = yaml.safe_load(open(cfg_path))
    dataset = RGBINRDataset(cfg['DATA'],  img_id)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    data = next(enumerate(dataloader))
    # img = data[1]['gt_img'].view(dataset.sidelength[1], dataset.sidelength[0], dataset.img_channels).numpy()
    img = dataset.img
    segmentation = data[1]['segments'].view(dataset.sidelength[1], dataset.sidelength[0], -1).numpy()
    # plt.imshow(img, cmap='gray')

    plt.imshow(img)
    # plt.imshow(dataset.freq_domain, cmap='gray')
    # plot segments with a discrete colormap
    # plt.imshow(segmentation, cmap='tab20', alpha=0.5)
    plt.show()
