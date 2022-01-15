
import argparse

def train_options():
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    
    parser.add_argument('--save_root', type=str, default='../results')
    parser.add_argument('--ckpt_id', type=str, default='pix2pix_testing')
    parser.add_argument('--dataset', default = 'dataset/facades', help='facades')
    parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
    parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
    
    return parser.parse_args()