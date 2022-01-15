import torch
import torch.nn as nn

from utils import utils
import time

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = torch.tensor(target_real_label).cuda()
        self.fake_label = torch.tensor(target_fake_label).cuda()

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)

        return self.loss(input, target_tensor)
    
class lossCollector():
    def __init__(self,args):
        self.args = args
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()
        self.GANL = GANLoss()
        self.loss_dict = {}
        self.start_time = time.time()

    def get_L1_loss(self,a,b):
        return self.L1(a, b)

    def get_L2_loss(self,a,b):
        return self.L2(a, b)

    def get_GAN_loss(self, a, target_is_real):
        return self.GANL(a, target_is_real)

    def get_D_loss(self,pred_fake,pred_real):
        loss_d_fake = self.get_GAN_loss(pred_fake,False)
        loss_d_real = self.get_GAN_loss(pred_real,True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        self.loss_dict['L_D'] = round(loss_d.item(),4)
        self.loss_dict['L_fake'] = round(loss_d.item(),4)
        self.loss_dict['L_real'] = round(loss_d.item(),4)
        return loss_d

    def get_G_loss(self, pred_fake, fake_b, real_b):
        loss_g_gan = self.get_GAN_loss(pred_fake,True)
        loss_g_l1 = self.get_L1_loss(fake_b,real_b) * self.args.lamb
        loss_g = loss_g_gan + loss_g_l1

        self.loss_dict['L_gan'] = round(loss_g_gan.item(),4)
        self.loss_dict['L_l1'] = round(loss_g_l1.item(),4)
        self.loss_dict['L_G'] = round(loss_g.item(),4)
        
        return loss_g

    def print_loss(self, global_step):

        seconds = int(time.time() - self.start_time)
        print("")
        print(f"[ {seconds//3600//24:02}d {(seconds//3600)%24:02}h {(seconds//60)%60:02}m {seconds%60:02}s ]")
        print(f'steps: {global_step:06} / {self.args.max_step}')
        print(f'lossD: {self.loss_dict["L_D"]} | lossG: {self.loss_dict["L_G"]}')
    