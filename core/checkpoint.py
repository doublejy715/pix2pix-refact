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

    def load_ckpt(self, G, D, opt_G, opt_D):
        try:
            # set path
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            
            # load ckpt
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
            
            # load state dict
            G.load_state_dict(ckpt["G"], strict=False)
            D.load_state_dict(ckpt["D"], strict=False)
            opt_G.load_state_dict(ckpt["opt_G"], strict=False)
            opt_D.load_state_dict(ckpt["opt_D"], strict=False)

        except Exception as e:
            print(e)

    def save_ckpt(self, global_step, G, D, opt_G, opt_D):
        os.makedirs(f'{self.args.save_root}/{self.args.run_id}/ckpt', exist_ok=True)

        ckpt_dict = {
            "G": G,
            "D": D,
            "opt_G": opt_G,
            "opt_D": opt_D,
        }

        ckpt_path = f'{self.args.save_root}/{self.args.run_id}/ckpt/{global_step}.pt'
        torch.save(ckpt_dict, ckpt_path)

        ckpt_path_latest = f'{self.args.save_root}/{self.args.run_id}/ckpt/latest.pt'
        torch.save(ckpt_dict, ckpt_path_latest)
        