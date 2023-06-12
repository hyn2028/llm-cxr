from pathlib import Path
from tkinter import filedialog

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from omegaconf import OmegaConf
from PIL import Image

from taming.models.vqgan import GumbelVQ, VQModel

    
def load_image(path, target_image_size=256):
    img = Image.open(path)

    if not img.mode == "RGB":
        img = img.convert("RGB")

    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = T.ToTensor()(img)
    img = 2.*img - 1.

    return img

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
RESIZE_SIZE = 256

torch.set_grad_enabled(False)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_path = Path(filedialog.askopenfilename(title="Select load path", defaultextension=".png"))

config = load_config(PATH_CONFIG, display=False)
model = load_vqgan(config, ckpt_path=PATH_CKPT, is_gumbel=False).to(DEVICE)

img = load_image(img_path, RESIZE_SIZE).to(DEVICE)
_, _, [_, _, indices] = model.encode(img.unsqueeze(0))
indices = indices.reshape(1, -1).cpu().squeeze().tolist()
print(indices)