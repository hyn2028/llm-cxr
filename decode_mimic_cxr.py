import torch
import numpy as np
from omegaconf import OmegaConf
import yaml
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path
from tkinter import filedialog

from taming.models.vqgan import GumbelVQ, VQModel

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

PATH_CONFIG = "<path to the trained model config (.yaml)>"
PATH_CKPT = "<path to the trained model ckpts (.ckpt)>"

torch.set_grad_enabled(False)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_path = Path(filedialog.asksaveasfilename(title="Select save path", defaultextension=".png"))

config = load_config(PATH_CONFIG, display=False)
model = load_vqgan(config, ckpt_path=PATH_CKPT, is_gumbel=False).to(DEVICE)

indices = eval(input(">> "))

assert len(indices) == 256
indices = torch.tensor(indices).to(DEVICE)

img = model.decode(model.quantize.get_codebook_entry(indices, shape=(1, 16, 16, -1)))
img = img.squeeze().permute(1,2,0).cpu().numpy()
img = np.clip(img, -1., 1.)
img = (img + 1.)/2.

plt.imsave(save_path, img)
