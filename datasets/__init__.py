import torch
from torch.utils.data import DataLoader
from .RGBImages import RGBINRDataset
from .dataset3dSDF import SDFdataset3D
from .Audio import AudioINRDataset

def build_dataset(cfg, file_id, training=True):
    if cfg['DATA'].get('name') == 'RGBINR':
        dataset = RGBINRDataset(cfg['DATA'], file_id)
    elif cfg['DATA'].get('name') == 'sdf_3d':
        dataset = SDFdataset3D(cfg, file_id)
    elif cfg['DATA'].get('name') == 'Audio':
        dataset = AudioINRDataset(cfg['DATA'], file_id)
    else:
        raise NotImplementedError
    return dataset


def build_dataloader(cfg, file_id, training=True):
    split = 'TRAINING' if training else 'TESTING'
    dataset = build_dataset(cfg, file_id, training)
    dataloader = DataLoader(dataset=dataset, num_workers=0, pin_memory=False,
                            batch_size=cfg[split]['batch_size'], shuffle=False)

    return dataloader, dataset