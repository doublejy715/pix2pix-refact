
import argparse

def train_options():
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    # 필수 args

    # roots
    parser.add_argument('--dataset', default='datasets/facades')
    parser.add_argument('--save_root', type=str, default='results') # pre-trained neural network

    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_step', type=int, default=200000)
    parser.add_argument('--adv_loss_type', type=str, default="lsgan")

    # ids
    parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--run_id', type=str, required=True) 
    parser.add_argument('--ckpt_id', type=str, default=None)
    parser.add_argument('--project_id', type=str, default="pix2pix")

    # log cycle
    parser.add_argument('--image_cycle', type=int, default=100)
    parser.add_argument('--loss_cycle', type=int, default=10)
    parser.add_argument('--ckpt_cycle', type=int, default=10000)

    # optimizer
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    
    # weight of losses
    # generator: required
    # discriminator: not required
    parser.add_argument('--W_adv', type=int, default=10, help='weight on adv term in objective')
    parser.add_argument('--W_L1', type=int, default=10, help='weight on L1 term in objective')
    parser.add_argument('--W_L2', type=int, default=0, help='weight on L2 term in objective')

    # etc
    parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--num_works', type=int, default=16, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use')
    parser.add_argument('--isMaster', default=False)
    parser.add_argument('--use_mGPU', action='store_false')

    return parser.parse_args()