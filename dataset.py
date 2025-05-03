import os

from PIL import Image
from glob import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import cv2
import numpy as np
from random import randint

class InitDataset(Dataset):
    def __init__(self, data_root, img_transform, mask_transform, data='train'):
        super(InitDataset, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        if data == 'train':
            self.paths = glob('{}/train/**/*.jpg'.format(data_root),
                              recursive=True)
            self.mask_paths = glob('{}/mask/*.png'.format(data_root))
        else:
            self.paths = glob('{}/test/*.jpg'.format(data_root, data))
            self.mask_paths = glob('{}/mask/*.png'.format(data_root))

        self.N_mask = len(self.mask_paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self._load_img(self.paths[index])
        img = self.img_transform(img.convert('RGB'))
        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return img * mask, mask, img

    def _load_img(self, path):
        """
        For dealing with the error of loading image which is occured by the loaded image has no data.
        """
        try:
            img = Image.open(path)
        except:
            extension = path.split('.')[-1]
            for i in range(10):
                new_path = path.split('.')[0][:-1] + str(i) + '.' + extension
                try:
                    img = Image.open(new_path)
                    break
                except:
                    continue
        return img

class MaskGenerator(object):

    def __init__(self, height, width, channels=3,
                 filepath=None):
        """Convenience functions for generating masks to be used for inpainting training
        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width
        Keyword Arguments:
            channels {int} -- Channels to output (default: {3})
            filepath {[type]} -- Load masks from filepath. If None, generate masks with OpenCV (default: {None})
        """

        self.height = height
        self.width = width
        self.channels = channels
        self.filepath = filepath

        # If filepath supplied, load the list of masks within the directory
        self.mask_files = []
        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames
                               if any(filetype in f.lower()
                                      for filetype
                                      in ['.jpeg', '.png', '.jpg'])]
            print("Found {} masks in {}".format(len(self.mask_files),
                                                   self.filepath))

    def _generate_mask(self):
        """Generates a random irregular mask with lines, circles and elipses"""

        img = np.zeros((self.height, self.width, self.channels), np.uint8)

        # Set size scale
        size = int((self.width + self.height) * 0.015)
        if self.width < 64 or self.height < 64:
            raise Exception("Width and Height of mask must be at least 64!")

        # Draw random lines
        for _ in range(randint(1, 20)):
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            thickness = randint(3, size)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        # Draw random circles
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(3, size)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        # Draw random ellipses
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            s1, s2 = randint(1, self.width), randint(1, self.height)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3,
                        (1, 1, 1), thickness)

        #Draw random rectangles
        for _ in range(randint(0, 2)):
            x1, y1 = randint(1, self.width//5), randint(1, self.height//5)
            x2, y2 = randint(1, self.width//5), randint(1, self.height//5)
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            thickness = randint(3, max(4, size))  # Ensure valid thickness
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(1, 1, 1), thickness=-1)

        return 1 - img

    def _load_mask(self, rotation=True, dilation=True, cropping=True):
        """Loads a mask from disk, and optionally augments it"""

        # Read image
        mask = cv2.imread(os.path.join(self.filepath, np.random.choice(
                                                        self.mask_files,
                                                        1,
                                                        replace=False
                                                        )[0]))

        # Random rotation
        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2),
                                        rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        # Random dilation
        if dilation:
            rand = np.random.randint(5, 47)
            kernel = np.ones((rand, rand), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        # Random cropping
        if cropping:
            x = np.random.randint(0, mask.shape[1] - self.width)
            y = np.random.randint(0, mask.shape[0] - self.height)
            mask = mask[y:y+self.height, x:x+self.width]

        return (mask > 1).astype(np.uint8)

    def sample(self):
        """Retrieve a random mask"""
        #if self.filepath and len(self.mask_files) > 0:
        #    return self._load_mask()
        #else:
        return self._generate_mask()

def main():
  NUM_MASK = 1000

  DIR_NAME = './data_new/mask/'
  if os.path.exists(DIR_NAME):
    pass
  else:
    os.mkdir(DIR_NAME)

  mask_generator = MaskGenerator(256, 256, channels=3,filepath=None)

  for idx in range(NUM_MASK):
      mask = mask_generator.sample() * 255
      cv2.imwrite('{}/{}.png'.format(DIR_NAME, idx), mask)

if __name__ == '__main__':
    main()