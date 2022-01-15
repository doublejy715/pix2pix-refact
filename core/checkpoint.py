import torch
import os

class ckptIO():
    def __init__(self, args):
        super(ckptIO, self).__init__()
        self.args = args

    def test_load_ckpt(self):
        G_ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/G_latest.pth'
        G = torch.load(G_ckpt_path)

        return G
        
    def load_ckpt(self):
        try:
            # set path
            G_ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/G_latest.pth'
            D_ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/D_latest.pth'
            
            # load ckpt
            G = torch.load(G_ckpt_path, map_location=torch.device('cuda'))
            D = torch.load(D_ckpt_path, map_location=torch.device('cuda'))

            return G,D


        except Exception as e:
            print(e)

    def save_ckpt(self, global_step, G, D):
        os.makedirs(f'{self.args.save_root}/{self.args.run_id}/ckpt', exist_ok=True)
        
        G_ckpt_path = f'{self.args.save_root}/{self.args.run_id}/ckpt/G_{global_step}.pt'
        D_ckpt_path = f'{self.args.save_root}/{self.args.run_id}/ckpt/D_{global_step}.pt'
        torch.save(G, G_ckpt_path)
        torch.save(D, D_ckpt_path)

        G_ckpt_path_latest = f'{self.args.save_root}/{self.args.run_id}/ckpt/G_latest.pt'
        D_ckpt_path_latest = f'{self.args.save_root}/{self.args.run_id}/ckpt/D_latest.pt'
        torch.save(G, G_ckpt_path_latest)
        torch.save(D, D_ckpt_path_latest)
        