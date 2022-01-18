"""
LFW dataloading
"""
import argparse
import time
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.io import read_image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.transform = transform
        self.folders = os.listdir(path_to_folder)
        self.path = path_to_folder
        
    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        folder_path = os.path.join(self.path, self.folders[index])
        img_path = os.listdir(folder_path)[0]
        img_path = os.path.join(folder_path, img_path)
        img = Image.open(img_path)
        return self.transform(img)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]

    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='', type=str)
    parser.add_argument('-num_workers', default=None, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    if args.visualize_batch:
        print("visualizing...")
        # TODO: visualize a batch of images
        plt.rcParams["savefig.bbox"] = 'tight'
        imgs = [img for img in Subset(dataset, list(range(32)))]
        imgs = make_grid(imgs)
        show(imgs)
        plt.show()
        
    if args.get_timing:
        '''
        means = []
        stds = [] 
        for num_workers in [0, 1, 2, 4, 8]:
        #for num_workers in [0, 4]:
            dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)

            # lets do some repetitions
            res = [ ]
            for _ in range(5):
                start = time.time()
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx > 100:
                        break
                end = time.time()
                res.append(end - start)

            mean = np.mean(res)
            means.append(mean)
            std = np.std(res)
            stds.append(std)
            print(f"Timing: {mean}+-{std}")
        '''
        means = [1,2,3,4,5]
        stds = [0.1, 0.1, 0.1, 0.1, 0.1]
        plt.errorbar([0,1,2,4,8], means, stds)
        plt.xlabel("num workers")
        plt.ylabel("timings")
        plt.savefig("err_plot.png")        
        plt.show()
