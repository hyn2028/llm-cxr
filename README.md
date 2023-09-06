# LLM Itself Can Read and Generate CXR Images

This repository is the official implementation of the paper [LLM Itself Can Read and Generate CXR Images (arxiv)](https://arxiv.org/abs/2305.11490).

![llm2cxr](https://github.com/hyn2028/llm-cxr/assets/5181065/b0c12395-c003-4901-bde9-9ecfe8272079)

## Generation Example
For more generation examples, see the paper on [arxiv](https://arxiv.org/abs/2305.11490).
### Report-to-CXR
![cxr_gen](https://github.com/hyn2028/llm-cxr/assets/5181065/75cd2498-a2c6-4f40-9a38-356b465e01ad)
### CXR-to-Report 
![report_gen](https://github.com/hyn2028/llm-cxr/assets/5181065/e3bc49ec-328b-4e57-961a-375bf0ac88ac)


## Install Dependencies

```bash
pip install -r requirements.txt
```

## Prepare Data
1. MIMIC-CXR report: Download `mimic-cxr-reports.zip` file from the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) dataset, unzip it, and place the `files` directory into the `data/mimic-cxr-reports` directory. To access the `MIMIC-CXR` dataset, appropriate credentials are required.

1. Quantized latent vector of `MIMIC-CXR` images: Convert 256x256 images from the MIMIC-CXR dataset to quantized latent vectors using the pre-trained VQ-GAN encoder presented below. Create a dictionary that maps from dicom_id to the latent vector and save it as `mimiccxr_vqgan1024_res256_3e_codebook_indices.pickle` file. Put it in `data/` directory. The latent vector must be in the form of a list of integers. See the "Encode and Decode Images" section below for how to use the encoder.

1. Instruction following dataset: Download `databricks-dolly-15k.jsonl` file from [here](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and put it in `data/` directory. This is used to preserve the natural language instruction following ability.

## Train Model
### Stage 1
Run the shell script below for the stage 1 train.

```bash
./train_llmcxr_stage1.sh
```
### Stage 2
Run the shell script below for the stage 2 train. 
```bash
./tarin_llmcxr_stage2.sh
```
Before running, modify the environment variable `input_model` in the `train_llmcxr_stage2.sh` file to point to the checkpoint path of the model trained in stage1.

### Training Tips
The current settings are geared towards `NVIDIA A100 40GB x8`, but if you change the `DeepSpeed` settings, you can also train on smaller GPUs like `NVIDIA GeForceRTX 3090 24GB x2`. Please refer to the original [code base repository](https://github.com/databrickslabs/dolly) and change the `DeepSpeed` settings (`config/ds_z3_bf16_config.json`) accordingly. Also adjust the `--num_gpus` argument in the `train_llmcxr_stage*.sh` file to match the number of GPUs.

## Inference

Place the downloaded pretrained LLM checkpoint (llmcxr_checkpoint-v3-1e+v2-2e.pth) in folder named 'checkpoints'.
Place the vqgan config and checkpoint files (2023-05-11T23-37-27-project.yaml, vqgan_last3d.ckpt) in a folder named 'vqgan'.

```bash
# clone this repository and move into it
git clone https://github.com/hyn2028/llm-cxr.git
cd llm-cxr

# create a new conda environment (with python3.9) and activate it
conda create -n llmcxr python=3.9
conda activate llmcxr

# install pytorch, torchvision, and pytorch-cuda
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# install dependencies
pip install -r requirements.txt

# Run inference.py
python inference.py --model_path /path/to/llmcxr_checkpoint/
```

## Pretrained Models
We provide all checkpoints used in the paper to generate and evaluate results. See the paper for details regarding training of uploaded checkpoints.
### LLM
We provide the pre trained LLM models through all-stage training. Download from [here](https://drive.google.com/file/d/1jaFT0yVOt8jyOQdvAZ2wfmBOsGQOlmIV/view?usp=sharing).
### VQ-GAN
We provide the pretrained VQ-GAN models for the image encoder and decoder. Download from [here](https://drive.google.com/file/d/15wIky-wUEuNMrEljKh5wnhAFg8QLdx-M/view?usp=sharing).


## Acknowledgements
We thank the authors for their great work. 
- We were heavily inspired by [UniXGen](https://arxiv.org/abs/2302.12172) for how we encode images to create them bidirectionally with transformers.
- Our training pipeline was modified from the [Databrick's Dolly](https://github.com/databrickslabs/dolly). 
- We also thank [Taming Transformers](https://github.com/CompVis/taming-transformers) for providing the architecture of `VQ-GAN` as image encoder and decoder. 
