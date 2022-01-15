import torch
import torch.nn as nn

from utils import utils
import time

class lossCollector():
    def __init__(self,args):
        self.args = args
        self.start_time = time.time()

        # define losses
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()

        # define adversarial loss
        if args.adv_loss_type == 'lsgan':
            self.adv = nn.MSELoss()
        elif args.adv_loss_type == 'vanilla':
            self.adv = nn.BCELoss()

        # dictionary for loss
        self.loss_dict = {}

    def get_L1_loss(self,a,b):
        return self.L1(a, b)

    def get_L2_loss(self,a,b):
        return self.L2(a, b)

    def get_adv_loss(self, logit, label):
        if label:
            label_ = torch.ones_like(logit, device='cuda')
        else:
            label_ = torch.zeros_like(logit, device='cuda')
        return self.adv(logit, label_)

    def get_D_loss(self, pred_fake, pred_real):
        loss_d_fake = self.get_adv_loss(pred_fake, False)
        loss_d_real = self.get_adv_loss(pred_real, True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        self.loss_dict['L_D'] = round(loss_d.item(),4)
        self.loss_dict['L_fake'] = round(loss_d.item(),4)
        self.loss_dict['L_real'] = round(loss_d.item(),4)
        return loss_d

    def get_G_loss(self, pred_fake, fake_b, real_b):

        L_G = 0.0

        # adv_loss
        if self.args.W_adv:
            L_gan = self.get_adv_loss(pred_fake, True)
            L_G += self.args.W_adv * L_gan
            self.loss_dict['L_gan'] = round(L_gan.item(),4)
        
        # L1_loss
        if self.args.W_L1:
            L_L1 = self.get_L1_loss(fake_b, real_b)
            L_G += self.args.W_L1 * L_L1
            self.loss_dict['L_L1'] = round(L_L1.item(),4)
            
        # L2_loss
        if self.args.W_L2:
            L_L2 = self.get_L2_loss(fake_b, real_b)
            L_G += self.args.W_L2 * L_L2
            self.loss_dict['L_l2'] = round(L_L2.item(),4)

        self.loss_dict['L_G'] = round(L_G.item(),4)
        
        return L_G

    def print_loss(self, global_step):

        seconds = int(time.time() - self.start_time)
        print("")
        print(f"[ {seconds//3600//24:02}d {(seconds//3600)%24:02}h {(seconds//60)%60:02}m {seconds%60:02}s ]")
        print(f'steps: {global_step:06} / {self.args.max_step}')
        print(f'lossD: {self.loss_dict["L_D"]} | lossG: {self.loss_dict["L_G"]}')
    