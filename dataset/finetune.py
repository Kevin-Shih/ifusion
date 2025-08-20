
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
    def __init__(self, image_dir, transform_fp):
        self.setup(image_dir, transform_fp)

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, index):
        index_target, index_cond = (
            self.perm[index, 0].item(),
            self.perm[index, 1].item(),
        )
        # rand scenes index -> batch size scene
        # rand ids index -> 1 for all scene (2 index, 1 for ref., 1 for target)
        # run model from another ids (same scene, same target) to get the loss
        scene_idxs = torch.randint(0, len(scenes), (args.batch_size,), device=model.device)
        img_idxs, next_img_idxs = torch.randint(0, len(ids), (2), device=model.device)
        next_img_idxs[1] = img_idxs[1]  # make sure the second image has the same idx for all scenes
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