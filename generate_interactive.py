from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from transformers import PreTrainedTokenizer
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
import gradio as gr

from training.generate import generate_response, load_model_tokenizer_for_generate
from training.mimiccxr_vq_dataset import CXR_VQ_TOKENIZER_LEN

import argparse
import numpy as np
from typing import List
from pathlib import Path


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=Path,
                        help='Path to LLM-CXR model checkpoint.')
    parser.add_argument('vqgan_config_path', type=Path,
                        help='Path to VQGAN config.')
    parser.add_argument('vqgan_ckpt_path', type=Path,
                        help='Path to VQGAN checkpoint.')
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
    img = TF.resize(img, s, interpolation=TF.InterpolationMode.LANCZOS)
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
    return response, None

def inference_t2i_t2t(model, tokenizer, vqgan, instruction, cxr_text_report):
    response, response_vq = generate_response(
        (instruction, cxr_text_report),
        model=model, tokenizer=tokenizer, max_new_tokens=512
    )

    if response_vq is not None:
        indices = torch.tensor(response_vq)
        quant = vqgan.quantize.get_codebook_entry(indices, shape=(1,16,16,-1))
        img = vqgan.decode(quant)
        img = img.squeeze().permute(1,2,0).detach().cpu().numpy()
        img = np.clip(img, -1., 1.)
        img = (img + 1.) / 2.
    else:
        img = None
    
    return response, img
    

if __name__ == '__main__':
    
    args = get_arguments()
    
    print("Loading model and tokenizer... Watch the system and GPU RAM")
    model, tokenizer = load_model_tokenizer_for_generate(args.model_path)
    print("LLM-CXR loaded!")
    
    print("Loading VQGAN for image encoding/decoding")
    vqgan = load_vqgan(args.vqgan_config_path, args.vqgan_ckpt_path)
    print("VQGAN loaded!")

    def gradio_interface(instruction, text_input, image_input):
        if instruction == "":
            return gr.Error("Please enter an instruction/question.")

        if image_input is not None:
            if text_input != "":
                return gr.Error("Please enter either an image or text, not both.")
            
            return inference_i2t(model, tokenizer, vqgan, instruction, image_input)
        else:
            return inference_t2i_t2t(model, tokenizer, vqgan, instruction, text_input)
    
    demo = gr.Interface(title="LLM-CXR", 
                        fn=gradio_interface, 
                        inputs=[gr.Textbox(label="Instruction"), gr.Textbox(label="Text input"), gr.Image(label="CXR input", type="filepath")], 
                        outputs=[gr.Textbox(label="Text response"), gr.Image(label="CXR response", type="numpy")], 
                        allow_flagging="never",
                        description="LLM-CXR is a CXR-report bidirectional generation LLM that can perform CXR-to-report, report-to-CXR, CXR vision question answering, and natural instruction following tasks. Either CXR input or Text input must be provided. Due to the specifications of the dataset, CXR input examples are not given. Please input the MIMIC-CXR-JPG dataset.",
                        examples=[["Generate a chest X-ray image that corresponds to the entered free-text radiology reports for the chest X-ray image.", "Moderate-sized right-sided pleural effusion with associated compressive atelectasis of the right lung. The left lung is clear."], 
                                  ["Generate free-text radiology reports for the entered chest X-ray images.", ""],
                                  ["Does the patient have lung opacity?", ""],
                                  ["What is  deep learning?", ""]],
                        cache_examples=False)
    demo.launch()