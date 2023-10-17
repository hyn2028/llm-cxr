import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from taming.models.vqgan import VQModel

import yaml
import pickle
from pathlib import Path
import argparse


class EvalDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            self.sampled = pickle.load(f)
            print(f"Total {len(self.sampled)} samples.")

            self.sampled = [x for x in self.sampled if not is_contain_none(x)]
            print(f"Total {len(self.sampled)} samples after removing None.")
            
    def __len__(self):
        return len(self.sampled)
    
    def __getitem__(self, index):
        sample = self.sampled[index]
        did = sample["dicom_id"]
        raw_image = torch.tensor(sample["raw_image"])
        gen_image = torch.tensor(sample["gen_image"])

        return {"raw_image": raw_image, "gen_image": gen_image, "dicom_id": did}


def is_contain_none(x: dict):
    for k, v in x.items():
        if v is None:
            return True
    return False

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
    model.load_state_dict(sd, strict=True)
    return model.eval()

def nn_img_to_img(img):
    img = img.permute(0, 2, 3, 1).cpu().numpy()
    img = np.clip(img, -1., 1.)
    img = (img + 1.)/2.
    return img

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('vqgan_config_path', type=Path,
                        help='Path to VQGAN config.')
    parser.add_argument('vqgan_ckpt_path', type=Path,
                        help='Path to VQGAN checkpoint.')
    parser.add_argument("path_result", type=Path,
                        help="Path to save result.")
    parser.add_argument("path_input", type=Path,
                        help="Path to generation result pickle.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_arguments()

    config = load_config(args.vqgan_config_path, display=False)
    model = load_vqgan(config, ckpt_path=args.vqgan_ckpt_path, is_gumbel=False).to(DEVICE)

    dataset = EvalDataset(args.path_input)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    dicom_ids = []
    gen_imgs = []
    for sample in tqdm(dataloader):
        gen_image = sample["gen_image"].to(DEVICE)
        raw_image = sample["raw_image"].to(DEVICE)
        dicom_id = sample["dicom_id"]

        B = gen_image.shape[0]

        gen_img = model.decode(model.quantize.get_codebook_entry(gen_image.flatten(), shape=(B, 16, 16, -1)))
        gen_img = nn_img_to_img(gen_img)

        dicom_ids.extend(dicom_id)
        gen_imgs.extend(gen_img)

    assert len(dicom_ids) == len(gen_imgs)
    assert len(dicom_ids) == len(dataset)

    (args.path_result / "generated_imgs_jpg").mkdir(parents=True, exist_ok=False)
    for dicom_id, gen_img in zip(tqdm(dicom_ids), gen_imgs):
        plt.imsave(args.path_result / "generated_imgs_jpg" / f"{dicom_id}.jpg", gen_img)

    with open(args.path_result / "generated_reports", "w") as f:
        for sample in tqdm(dataset.sampled):
            dicom_id = sample["dicom_id"]
            raw_report = sample["raw_report"]
            gen_report = sample["gen_report"]
            f.write(f"{dicom_id}\n{raw_report}\n{gen_report}\n\n")
            