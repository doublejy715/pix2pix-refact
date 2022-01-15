
import argparse

def train_options():
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    # 필수 args
    parser.add_argument('--save_root', type=str, default='ptnn')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--use_mGPU', action='store_false')
    parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--run_id', type=str, required=True) 
    parser.add_argument('--isMaster', default=False)
    parser.add_argument('--max_step', type=int, default=200000)
    parser.add_argument('--image_cycle', type=int, default=100)
    parser.add_argument('--loss_cycle', type=int, default=10)
    parser.add_argument('--ckpt_cycle', type=int, default=10000)
    parser.add_argument('--ckpt_id', type=str, default=None)

    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--dataset', default='facades')# required=True, help='facades')
    parser.add_argument('--root_path', default='datasets/facades', help='datasets/facades')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
    parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')

    return parser.parse_args()