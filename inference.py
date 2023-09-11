#%%
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from training.generate import generate_response, load_model_tokenizer_for_generate
from training.mimiccxr_vq_dataset import sample_cxr_vq_output_instruction, sample_cxr_vq_input_instruction, CXR_VQ_TOKENIZER_LEN

from transformers import PreTrainedTokenizer

from typing import List
from argparse import ArgumentParser

from omegaconf import OmegaConf
from taming.models.vqgan import VQModel


#%%
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/llmcxr_checkpoint-v3-1e+v2-2e',
                        help='path to llm-cxr model checkpoint')
    parser.add_argument('--vqgan_config_path', type=str,
                        default='vqgan/2023-05-11T23-37-27-project.yaml')
    parser.add_argument('--vqgan_ckpt_path', type=str,
                        default='vqgan/vqgan_last3e.ckpt')
    args = parser.parse_args()
    return args

def shift_vq_tokens(tokens: List[int], tokenizer: PreTrainedTokenizer) -> List[int]:
    assert len(tokenizer) == CXR_VQ_TOKENIZER_LEN
    return [token + len(tokenizer) for token in tokens]

def load_vqgan(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    vqgan = VQModel(**config.model.params)
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    vqgan.load_state_dict(state_dict, strict=False)
    vqgan.eval()
    return vqgan

def inference_i2t(model, tokenizer, vqgan, instruction, cxr_image_path):
    img = Image.open(cxr_image_path).convert('RGB')
    s = min(img.size)
    target_image_size = 256
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 *[target_image_size])
    img = T.ToTensor()(img)
    img = 2. * img - 1.
    
    # Get latent representation (ie VQ-encoding)
    z,_, [_, _, indices] = vqgan.encode(img.unsqueeze(0))
    img_vq = shift_vq_tokens(indices, tokenizer)
    
    response, response_vq = generate_response(
        (instruction, img_vq), model=model, tokenizer=tokenizer,
        max_new_tokens=512
    )
    return response

def inference_t2i(model, tokenizer, vqgan, instruction, cxr_text_report):
    response, response_vq = generate_response(
        (instruction, cxr_text_report),
        model=model, tokenizer=tokenizer, max_new_tokens=512
    )
    indices = torch.tensor(response_vq)
    quant = vqgan.quantize.get_codebook_entry(indices, shape=(1,16,16,-1))
    img = vqgan.decode(quant)
    img = img.squeeze().permute(1,2,0).detach().cpu().numpy()
    img = np.clip(img, -1., 1.)
    img = (img + 1.) / 2.
    
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.set_axis_off()
    for item in [fig, ax]:
        item.patch.set_visible(False)
    ax.imshow(img)
    plt.show()
    
def inference_qna(model, tokenizer, vqgan, instruction):
    response, _ = generate_response(
        (instruction, None),
        model=model, tokenizer=tokenizer, max_new_tokens=512
    )
    return response
        
#%%
if __name__ == '__main__':
    
    args = get_arguments()
    
    print("Loading model and tokenizer... Watch the system and GPU RAM")
    model, tokenizer = load_model_tokenizer_for_generate(args.model_path)
    print("LLM-CXR loaded!")
    
    print("Loading VQGAN for image encoding/decoding")
    vqgan = load_vqgan(args.vqgan_config_path, args.vqgan_ckpt_path)
    print("VQGAN loaded!")
    
    while True:
        cxr_image_path = input("\nIf you are inputting a CXR image, type its path now. Otherwise, prese enter.\n------------------------------------------------------------\n")
        cxr_text_report = input("\nIf you are inputting a text report for a CXR, type it in now. Otherwise, press enter.\n------------------------------------------------------------\n")
        instruction = input("\nWhat is your instruction/question?\n------------------------------------------------------------\n")
        
        # Image-to-Text (ie, CXR-to-Report)
        if os.path.isfile(cxr_image_path) and len(cxr_text_report)==0:
            response = inference_i2t(model, tokenizer, vqgan, instruction, cxr_image_path)
            print(response)
        
        # Text-to-Image (ie, Report-to-CXR)
        elif len(cxr_image_path)==0 and len(cxr_text_report) >= 1:
            inference_t2i(model, tokenizer, vqgan, instruction, cxr_text_report)
        
        # QNA
        elif not os.path.isfile(cxr_image_path) and len(cxr_text_report)==0:
            response = inference_qna(model, tokenizer, vqgan, instruction)
            print(response)
        
        else:
            print("\nInvalid input. Please query again.\n------------------------------------------------------------\n")
            
        inp = input("\nPress q to quit. Press any other key to try another example.\n------------------------------------------------------------\n")
        print(inp)
        if inp == "q":
            break