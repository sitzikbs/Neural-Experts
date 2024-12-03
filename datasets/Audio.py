from torch.utils.data import Dataset
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import os
import json

class AudioINRDataset(Dataset):
    def __init__(self, data_cfg, audiofile_id):
        super().__init__()
        self.data_cfg = data_cfg
        self.audiofile_id = audiofile_id
        self.rate, self.data = wavfile.read(os.path.join(data_cfg['dataset_path'], audiofile_id))
        if len(self.data.shape) > 1 and self.data.shape[1] == 2: # stereo / mono
            self.data = np.mean(self.data, axis=1)
        self.data = self.data.astype(np.float32)
        print("Rate: %d" % self.rate)
        self.sidelength = len(self.data)
        self.grid = np.linspace(start=-100, stop=100, num=self.sidelength)
        self.grid = self.grid.astype(np.float32)
        self.grid = torch.Tensor(self.grid).view(-1, 1)

        self.n_segments = data_cfg['n_segments']
        self.segmentation_type = data_cfg['segmentation_type']
        if self.segmentation_type == 'random_balanced':
            self.segments = self.random_balanced_segmentation()
        elif self.segmentation_type == 'gt':
            self.segments = self.gt_segmentation()
        else:
            raise ValueError("Segmentation type not recognized: %s" % self.segmentation_type)

    def random_balanced_segmentation(self):
        ''' Create a random segmentation of rows and columns.'''
        segments = np.random.randint(0, self.n_segments, len(self.data))
        return segments[None, :].flatten()[None, :]

    def gt_segmentation(self):
        segmentation_filename = os.path.join(self.data_cfg['dataset_path'], self.audiofile_id).replace('.wav', '_segments.json')
        if not os.path.exists(segmentation_filename):
            raise ValueError("Segmentation file not found: %s" % segmentation_filename)
        else:
            with open(segmentation_filename) as f:
                segments_data = json.load(f)

        segments = np.zeros(len(self.data))
        for seg in segments_data:
            segments[int(seg['start']*self.rate):int(seg['end']*self.rate)] = seg['label']

        self.n_segments = len(np.unique(segments))
        return segments[None, :].flatten()[None, :].astype(int)

    def get_num_samples(self):
        return self.grid.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        rate, data = self.rate, self.data
        scale = np.max(np.abs(data))
        data = (data / scale)
        data = torch.Tensor(data).view(-1, 1)
        segments = self.segments

        out_dict = {'coords': self.grid, 'gt_data': data, 'segments': segments, 'rate': rate, 'scale': scale}
        return out_dict


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import utils.visualizations as vis
    import matplotlib.pyplot as plt


    data_cfg = {'n_segments': 10, 'dataset_path': '../data/audio/', 'segmentation_type': 'gt'}
    audiofile_id = 'gt_twospeakers.wav'
    dataset = AudioINRDataset(data_cfg, audiofile_id)
    # img = vis.plot_audio_waveform_w_segments(dataset.grid, dataset.data, dataset.segments)
    vis.plot_audio_spectrograms_w_segments(dataset.grid, dataset.data, dataset.rate, dataset.segments)
    # plt.imshow(img)
    # plt.show()
    print(dataset[0]['gt_data'].shape)
    print(dataset[0]['coords'].shape)
    print(dataset[0]['segments'].shape)
    print(dataset[0]['rate'])
    print(dataset[0]['scale'])
    print(dataset.get_num_samples())
