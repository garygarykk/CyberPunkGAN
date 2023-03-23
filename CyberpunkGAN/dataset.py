import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from GAN.image_processing import normalize_input, compute_data_mean

class NeonDataSet(Dataset):
    def __init__(self, args, transform=None):
        data_dir = args.data_dir
        dataset = args.dataset

        neon_dir = os.path.join(data_dir, dataset)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'Folder {data_dir} does not exist')

        if not os.path.exists(neon_dir):
            raise FileNotFoundError(f'Folder {neon_dir} does not exist')

        self.mean = compute_data_mean(os.path.join(neon_dir, 'style'))
        print(f'Mean(B, G, R) of {dataset} are {self.mean}')

        self.debug_samples = args.debug_samples or 0
        self.data_dir = data_dir
        self.image_files =  {}
        self.photo = 'real_img'
        self.style = f'{neon_dir}/style'
        self.smooth =  f'{neon_dir}/smooth'
        self.dummy = torch.zeros(3, 256, 256)

        for opt in [self.photo, self.style, self.smooth]:
            folder = os.path.join(data_dir, opt)
            files = os.listdir(folder)

            self.image_files[opt] = [os.path.join(folder, fi) for fi in files]

        self.transform = transform

        print(f'Dataset: real {len(self.image_files[self.photo])} style {self.len_neon}, smooth {self.len_smooth}')

    def __len__(self):
        return self.debug_samples or len(self.image_files[self.photo])

    @property
    def len_neon(self):
        return len(self.image_files[self.style])

    @property
    def len_smooth(self):
        return len(self.image_files[self.smooth])

    def __getitem__(self, index):
        image = self.load_photo(index)
        neon_idx = index
        if neon_idx > self.len_neon - 1:
            neon_idx -= self.len_neon * (index // self.len_neon)

        anime, anime_gray = self.load_anime(neon_idx)
        smooth_gray = self.load_anime_smooth(neon_idx)

        return image, anime, anime_gray, smooth_gray

    def load_photo(self, index):
        fpath = self.image_files[self.photo][index]
        image = cv2.imread(fpath)[:,:,::-1]
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)

    def load_anime(self, index):
        fpath = self.image_files[self.style][index]
        image = cv2.imread(fpath)[:,:,::-1]

        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
        image_gray = self._transform(image_gray, addmean=False)
        image_gray = image_gray.transpose(2, 0, 1)

        image = self._transform(image, addmean=True)
        image = image.transpose(2, 0, 1)

        return torch.tensor(image), torch.tensor(image_gray)

    def load_anime_smooth(self, index):
        fpath = self.image_files[self.smooth][index]
        image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)

    def _transform(self, img, addmean=True):
        if self.transform is not None:
            img =  self.transform(image=img)['image']

        img = img.astype(np.float32)
        if addmean:
            img += self.mean
    
        return normalize_input(img)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    neon_loader = DataLoader(NeonDataSet('TRAINING/neon_img/smooth'), batch_size=2, shuffle=True)

    img, img_gray = iter(neon_loader).next()
    plt.imshow(img[1].numpy().transpose(1, 2, 0))
    plt.show()
