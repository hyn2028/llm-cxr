import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from taming.models.vqgan import VQModel

import pickle
from pathlib import Path
import argparse


class ImageDataset(Dataset):
    def __init__(self, paths, target_image_size):
        self.target_image_size = target_image_size

        self.target_paths = []
        for path in paths:
            with open(path, "r") as f:
                self.target_paths.extend(Path(line.strip()) for line in f.readlines())

    def __len__(self):
        return len(self.target_paths)
    
    def __getitem__(self, index):
        path = self.target_paths[index]
        img = Image.open(path)

        if not img.mode == "RGB":
            img = img.convert("RGB")

        s = min(img.size)
        
        if s < self.target_image_size:
            raise ValueError(f'min dim for image {s} < {self.target_image_size}')
            
        r = self.target_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [self.target_image_size])
        img = T.ToTensor()(img)
        img = 2.*img - 1.

        return {"image": img, "dicom_id": path.stem}


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        raise NotImplementedError("GumbelVQ is not implemented yet.")
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.eval()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('vqgan_config_path', type=Path,
                        help='Path to VQGAN config.')
    parser.add_argument('vqgan_ckpt_path', type=Path,
                        help='Path to VQGAN checkpoint.')
    parser.add_argument("path_result", type=Path,
                        help="Path to save result.")
    parser.add_argument("paths_data_list", type=Path, nargs="+",
                        help="Path to data list.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    RESIZE_SIZE = 256
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_arguments()
    torch.set_grad_enabled(False)

    config = load_config(args.vqgan_config_path, display=False)
    model = load_vqgan(config, ckpt_path=args.vqgan_ckpt_path, is_gumbel=False).to(DEVICE)
    dataset = ImageDataset(args.paths_data_list, RESIZE_SIZE)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    dicomid_vq_latent_map = {}
    for batch in tqdm(dataloader, unit="batch", smoothing=0.0):
        img_batch = batch["image"].to(DEVICE)
        dicomid_batch = batch["dicom_id"]
        B = img_batch.shape[0]

        _, _, [_, _, indices_batch] = model.encode(img_batch)
        indices_batch = indices_batch.reshape(B, -1).cpu().tolist()
        
        dicomid_vq_latent_map.update(zip(dicomid_batch, indices_batch))

    print(f"target len: {len(dataset.target_paths)}")
    print(f"result len: {len(dicomid_vq_latent_map)}")
    with open(args.path_result, "wb") as f:
        pickle.dump(dicomid_vq_latent_map, f)