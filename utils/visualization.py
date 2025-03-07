from typing import List, Tuple
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

import logging

matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.INFO)


def scatter_drawer(pos: torch.Tensor, fix_mask: torch.Tensor, filename, title, args):
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path = os.path.join(res_root, args.eval_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))

    mov_pos = pos[fix_mask.squeeze(1) < 0.5].T.cpu().numpy()
    fix_pos = pos[fix_mask.squeeze(1) > 0.5].T.cpu().numpy()
    plt.scatter(mov_pos[0], mov_pos[1], label="mov")
    plt.scatter(fix_pos[0], fix_pos[1], label="fix")
    plt.legend()
    plt.title(title)
    plt.savefig(png_path)
    plt.close()

def draw_fig(fig, args, filename="fig.png"):
    png_path = os.path.join(args.res_root, args.eval_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))
    plt.figure()
    im = plt.imshow(fig.detach().cpu().numpy(), cmap="jet")
    rb = plt.colorbar(im)
    plt.savefig(png_path)
    plt.close()
    
def draw_fig_show(fig, res_root, filename="fig.png"):
    png_path = os.path.join(res_root, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))
    plt.figure()
    im = plt.imshow(np.rot90(fig), cmap="jet")
    rb = plt.colorbar(im)
    plt.savefig(png_path)
    plt.close()