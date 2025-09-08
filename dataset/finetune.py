
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from dataset.base import BaseDataset


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
        # rand scenes index -> batch size scene
        # rand ids index -> 1 for all scene (2 index, 1 for ref., 1 for target)
        # run model from another ids (same scene, same target) to get the loss
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

class MyFinetuneAllSceneIterableDataset(IterableDataset, MyFinetuneDataset):
    # def setup(self, image_dir, transform_fp):
    #     self.all_images, self.all_camtoworlds, _ = load_frames(image_dir, transform_fp)

    #     assert len(self.all_camtoworlds) == len(self.all_images)
    #     assert self.all_images.shape[2:] == (256, 256)

    #     self.perm = list(itertools.permutations(range(len(self.all_images)), 2))
    #     self.perm = torch.from_numpy(np.array(self.perm))
    # setup for whole eval
    pass
