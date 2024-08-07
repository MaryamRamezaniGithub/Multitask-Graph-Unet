from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import pandas as pd

import logging

logging.basicConfig(level=logging.DEBUG)



class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, csv_dir, scale=1):
    #def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.csv_dir = csv_dir
        self.orientation_data = pd.read_csv(csv_dir)

        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info('Creating dataset with %s examples'%(len(self.ids)))
        print('Creating dataset with %s examples'%(len(self.ids)))

        # print(self.ids)

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        # print("w , h", w , h)

        pil_img = pil_img.resize((192, 192))
        w, h = pil_img.size
        # print("w , h", w , h)

        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))


        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # if img_nd.max() > 1:
        #     img_nd = img_nd / 255

        # return img_nd

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        
        # print(idx, i, type(idx), type(i))
        # print("mask_file: ", mask_file)
        # print("img_file: ", img_file)

        
        orientation = self.orientation_data.iloc[int(idx), 1:4].to_numpy()
        orientation[0] = orientation[0]/100.0 - 0.3
        # print("distance: ", orientation[0])
        orientation = orientation.astype('float').reshape(-1, 1)

        # assert len(mask_file) == 1, \
        #     'Either no mask or multiple masks found for the ID %s: %s' %(idx,mask_file)

        # logging.debug(f"Index: {idx}, Image File: {img_file}")
        #print("image file: ", img_file)

        assert len(img_file) == 1, \
             'Either no image or multiple images found for the ID %s: %s'%(idx,img_file)
        mask = Image.open(mask_file[0]).convert('L')
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {idx} should be the same size, but are %s and %s'%(img.size, mask.size)

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), 'orientation': orientation}
        #return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

