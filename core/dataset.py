import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from os import listdir
from os.path import join
import random
from PIL import Image

import glob
from utils.utils import is_image_file, load_img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super(DatasetFromFolder, self).__init__()

        # direction
        self.direction = direction

        # image path from dir a 
        image_a_paths = sorted(glob.glob(f"{image_dir}/a/*.*"))
        self.image_a_paths = [x for x in image_a_paths if is_image_file(x)]

        # image path from dir b 
        image_b_paths = sorted(glob.glob(f"{image_dir}/b/*.*"))
        self.image_b_paths = [x for x in image_b_paths if is_image_file(x)]

        # horizontal flip, to tensor, normalize
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __getitem__(self, index):
        # load image
        a = Image.open(self.image_a_paths[index]).convert('RGB')
        b = Image.open(self.image_b_paths[index]).convert('RGB')

        # resize
        resize = transforms.Resize(size=(300, 300))
        a = resize(a)
        b = resize(b)

        # random crop
        box = transforms.RandomCrop.get_params(a, output_size=(256, 256)) 
        a = transforms.functional.crop(a, *box) 
        b = transforms.functional.crop(b, *box)

        # other transforms
        a = self.transform(a)
        b = self.transform(b)

        # direction
        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_b_paths)

