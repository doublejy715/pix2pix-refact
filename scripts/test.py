from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd()+"/..")

import torch
import torchvision.transforms as transforms

from utils.utils import is_image_file, load_img, save_img
from core.checkpoint import ckptIO
from core.test_options import train_options

if __name__ == "__main__":
    # Testing settings
    args = train_options()
    print(args)

    device = torch.device("cuda" if args.cuda else "cpu")

    # load model
    ckptio = ckptIO(args)
    G = ckptio.test_load_ckpt()
    G.to(device)

    if args.direction == "a2b":
        image_dir = "../datasets/{}/test/a/".format(args.dataset)
    else:
        image_dir = "../datasets/{}/test/b/".format(args.dataset)

    image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

    transform_list = [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)

    for image_name in image_filenames:
        img = load_img(image_dir + image_name)
        img = transform(img)
        input = img.unsqueeze(0).to(device)
        out = G(input)
        out_img = out.detach().squeeze(0).cpu()

        if not os.path.exists(os.path.join("../result", args.dataset)):
            os.makedirs(os.path.join("../result", args.dataset))
        save_img(out_img, "../result/{}/{}".format(args.dataset, image_name))