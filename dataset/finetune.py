
import torch
import itertools
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset

from dataset.base import BaseDataset, load_frames


class FinetuneDataset(Dataset, BaseDataset):
    def __init__(self, image_dir, transform_fp):
        self.setup(image_dir, transform_fp)

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, index):
        index_target, index_cond = (
            self.perm[index, 0].item(),
            self.perm[index, 1].item(),
        )
        return {
            "image_target": self.all_images[index_target],
            "image_cond": self.all_images[index_cond],
            "T": self.get_trans(self.all_camtoworlds[index_target], self.all_camtoworlds[index_cond], in_T=True),
        }

    def loader(self, batch_size=1, num_workers=8, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=None,
            **kwargs,
        )

class MyFinetuneDataset(Dataset, BaseDataset):
    def __init__(self,
                 image_dir,
                transform_fp: str = None,
    ):
        if 'scene' in transform_fp:
            print(f"[INFO] Load whole scene from {transform_fp}")
        else:
            print(f"[INFO] Load id[0,1] from {transform_fp}")
        self.setup(image_dir, transform_fp)

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, index):
        index_cond1, index_target = (
            self.perm[index, 0].item(),
            self.perm[index, 1].item(),
        )
        index2 = torch.randint(0, len(self.perm), size=(1,)).item()
        index_cond2 = self.perm[index2, 0].item()
        return {
            "image_target": self.all_images[index_target], # B, C, H, W
            "image_cond1": self.all_images[index_cond1], # B, C, H, W
            "image_cond2": self.all_images[index_cond2], # B, C, H, W
            "T1": self.get_trans(self.all_camtoworlds[index_target], self.all_camtoworlds[index_cond1], in_T=True), # B, 4 : [theta, torch.sin(azimuth), torch.cos(azimuth), distance]
            "T2": self.get_trans(self.all_camtoworlds[index_target], self.all_camtoworlds[index_cond2], in_T=True), # B, 4 : [theta, torch.sin(azimuth), torch.cos(azimuth), distance]
        }

    def loader(self, batch_size=1, num_workers=8, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=None,
            **kwargs,
        )

class MyFinetuneGeneralDataset(Dataset, BaseDataset):
    def __init__(self, num_scenes):
        self.perm = list(itertools.permutations(range(5), 3))
        self.perm = torch.from_numpy(np.array(self.perm))
        self.all_images = None
        self.all_camtoworlds = None
        self.num_scenes = num_scenes

    def __len__(self):
        return len(self.perm) * self.num_scenes

    def __getitem__(self, index):
        scene = index // len(self.perm)
        idx = index % len(self.perm)
        index_cond1, index_cond2, index_target = (
            self.perm[idx, 0].item(),
            self.perm[idx, 1].item(),
            self.perm[idx, 2].item(),
        )
        return {
            "image_target": self.all_images[scene, index_target], # B, C, H, W
            "image_cond1" : self.all_images[scene, index_cond1], # B, C, H, W
            "image_cond2" : self.all_images[scene, index_cond2], # B, C, H, W
            "T1": self.get_trans(self.all_camtoworlds[scene, index_target], self.all_camtoworlds[scene, index_cond1], in_T=True), # B, 4 : [theta, torch.sin(azimuth), torch.cos(azimuth), distance]
            "T2": self.get_trans(self.all_camtoworlds[scene, index_target], self.all_camtoworlds[scene, index_cond2], in_T=True), # B, 4 : [theta, torch.sin(azimuth), torch.cos(azimuth), distance]
        }

    def loader(self, batch_size=1, num_workers=8, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=None,
            **kwargs,
        )

class FinetuneIterableDataset(IterableDataset, FinetuneDataset):
    def __init__(self, image_dir, transform_fp):
        super().__init__(image_dir, transform_fp)

    def __iter__(self):
        while True:
            index = torch.randint(0, len(self.perm), size=(1,)).item()
            index_target, index_cond = (
                self.perm[index, 0].item(),
                self.perm[index, 1].item(),
            )
            yield {
                "image_target": self.all_images[index_target],
                "image_cond": self.all_images[index_cond],
                "T": self.get_trans(self.all_camtoworlds[index_target], self.all_camtoworlds[index_cond], in_T=True),
            }

class MyFinetuneIterableDataset(IterableDataset, MyFinetuneDataset):
    def __init__(self,
                 image_dir,
                 transform_fp: str = None,
    ):
        if 'scene' in transform_fp:
            print(f"[INFO] Load whole scene from {transform_fp}")
        else:
            print(f"[INFO] Load id[0,1] from {transform_fp}")
        self.setup(image_dir, transform_fp)

    def __iter__(self):
        while True:
            index1, index2 = torch.randint(0, len(self.perm), size=(2,))
            index_cond1, index_cond2, index_target = (
                self.perm[index1, 0].item(),
                self.perm[index2, 0].item(),
                self.perm[index1, 1].item(),
            )
            # print(f'index_cond1:{index_cond1}')
            yield {
                "image_target": self.all_images[index_target],
                "image_cond1": self.all_images[index_cond1],
                "image_cond2": self.all_images[index_cond2],
                "T1": self.get_trans(self.all_camtoworlds[index_target], self.all_camtoworlds[index_cond1], in_T=True),
                "T2": self.get_trans(self.all_camtoworlds[index_target], self.all_camtoworlds[index_cond2], in_T=True),
            }

class MyFinetuneAllSceneIterableDataset(IterableDataset, MyFinetuneGeneralDataset):
    def __init__(self, num_scenes):
        super().__init__(num_scenes)

    def add_scenes(self, image_dir, transform_fp):
        if 'scene' in transform_fp:
            print(f"[INFO] Load whole scene from {transform_fp}")
        else:
            print(f"[INFO] Load id[0,1] from {transform_fp}")
        images, camtoworlds, _ = load_frames(image_dir, transform_fp) # k, C, H, W and k, 4, 4

        assert len(images) == len(camtoworlds)
        assert images.shape[2:] == (256, 256)
        if self.all_images is None:
            self.all_images = images.unsqueeze(0) # S=1, k, C, H, W
            self.all_camtoworlds = camtoworlds.unsqueeze(0) # S=1, k, 4, 4
        else:
            self.all_images = torch.cat((self.all_images, images.unsqueeze(0))) # S+=1, k, C, H, W
            self.all_camtoworlds = torch.cat((self.all_camtoworlds, camtoworlds.unsqueeze(0))) # S+=1, k, 4, 4

    def __iter__(self):
        while True:
            scene = torch.randint(0, self.num_scenes, size=(1,)).item()
            index = torch.randint(0, len(self.perm), size=(1,)).item()
            index_cond1, index_cond2, index_target = (
                self.perm[index, 0].item(),
                self.perm[index, 1].item(),
                self.perm[index, 2].item(),
            )
            # print(f'index_cond1:{index_cond1}')
            yield {
                "image_target": self.all_images[scene, index_target],
                "image_cond1": self.all_images[scene, index_cond1],
                "image_cond2": self.all_images[scene, index_cond2],
                "T1": self.get_trans(self.all_camtoworlds[scene, index_target], self.all_camtoworlds[scene, index_cond1], in_T=True),
                "T2": self.get_trans(self.all_camtoworlds[scene, index_target], self.all_camtoworlds[scene, index_cond2], in_T=True),
            }