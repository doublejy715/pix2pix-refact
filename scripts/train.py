import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import sys
sys.path.append(os.getcwd())

from utils import utils
from core.checkpoint import ckptIO
from core.loss import lossCollector
from opts.train_options import train_options
from nets.model import define_G, define_D
from core.dataset import DatasetFromFolder

def train(gpu, args): 
    # set gpu
    torch.cuda.set_device(gpu)

    # build model
    G = define_G(gpu_id=gpu)
    D = define_D(netD_type='basic', gpu_id=gpu)
    
    # load and initialize the optimizer
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # load checkpoint
    ckptio = ckptIO(args)
    ckptio.load_ckpt(G, D, opt_G, opt_D)
    
    # build a dataset
    train_set = DatasetFromFolder(f"{args.dataset}/train", args.direction)
    test_set = DatasetFromFolder(f"{args.dataset}/test", args.direction)

    train_sampler = None
    test_sampler = None

    if args.use_mGPU:
        args.isMaster = gpu==0

        # DDP setup
        utils.setup_ddp(gpu, args.gpu_num)

        # Distributed Data Parallel
        G = torch.nn.parallel.DistributedDataParallel(G, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True).module
        D = torch.nn.parallel.DistributedDataParallel(D, device_ids=[gpu]).module

        # make sampler 
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # build a dataloader
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,sampler=train_sampler, num_workers=args.num_works,drop_last=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size,sampler=test_sampler, num_workers=args.num_works,drop_last=True)

    training_batch_iterator = iter(training_data_loader)
    testing_batch_iterator = iter(testing_data_loader)
    
    # build loss
    loss_collector = lossCollector(args)

    # initialize wandb
    if args.isMaster:
        wandb.init(project=args.project_id, name=args.run_id)

    global_step = -1
    while global_step < args.max_step:
        global_step += 1
        try:
            real_a, real_b = next(training_batch_iterator)
        except StopIteration:
            training_batch_iterator = iter(training_data_loader)
            real_a, real_b = next(training_batch_iterator)
        
        # transfer data to the gpus
        real_a, real_b = real_a.to(gpu), real_b.to(gpu)
        

        ###########
        # train G #
        ###########

        # run G
        fake_b = G(real_a)
        fake_ab = torch.cat((real_a, fake_b),1)
        pred_fake = D.forward(fake_ab)

        # calculate G Loss
        loss_G = loss_collector.get_G_loss(pred_fake, fake_b, real_b)

        # update G
        utils.update_net(opt_G, loss_G)

        ###########
        # train D #
        ###########

        # run D
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = D.forward(fake_ab.detach())

        # train with real
        real_ab = torch.cat((real_a, real_b),1)
        pred_real = D.forward(real_ab)
        
        # calculate D Loss
        loss_D = loss_collector.get_D_loss(pred_fake,pred_real)

        # update D
        utils.update_net(opt_D, loss_D)

        # log and print loss
        if args.isMaster and global_step % args.loss_cycle==0:
            
            # log loss on wandb
            wandb.log(loss_collector.loss_dict)
            
            # print loss
            loss_collector.print_loss(global_step)

        # save image
        if args.isMaster and global_step % args.image_cycle == 0:
            
            try:
                real_a, real_b = next(testing_batch_iterator)
            except StopIteration:
                testing_batch_iterator = iter(training_data_loader)
                real_a, real_b = next(testing_batch_iterator)

            real_a, real_b = real_a.to(gpu), real_b.to(gpu)
            fake_b = G(real_a)

            utils.save_image(args, global_step, "imgs", [real_a, real_b, fake_b])

        # save ckpt
        if global_step % args.ckpt_cycle == 0:
            ckptio.save_ckpt(global_step, G, D, opt_G, opt_D)

if __name__ == "__main__":
    
    # get args
    args = train_options()

    # make training dir
    os.makedirs(args.save_root, exist_ok=True)

    # setup multi-GPUs env
    if args.use_mGPU:

        # get gpu number
        args.gpu_num = torch.cuda.device_count()
        
        # divide by gpu num
        args.batch_size = int(args.batch_size / args.gpu_num)

        # start multi-GPUs training
        torch.multiprocessing.spawn(train, nprocs=args.gpu_num, args=(args, ))

    # if use single-GPU
    else:
        # set isMaster
        args.isMaster = True

        # start single-GPU training
        train(args.gpu_id, args)
