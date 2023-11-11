# LLM-CXR: Instruction-Finetuned LLM for CXR Image Understanding and Generation

This repository is the official implementation of the paper [LLM-CXR: Instruction-Finetuned LLM for CXR Image Understanding and Generation (arxiv)](https://arxiv.org/abs/2305.11490).

![llm2cxr](https://github.com/hyn2028/llm-cxr/assets/5181065/f2f3027e-1ab8-4fe5-bff6-82a4065176c9)

## Generation example
For more generation examples, see the paper on [arxiv](https://arxiv.org/abs/2305.11490).
### Report-to-CXR
![cxr_gen](https://github.com/hyn2028/llm-cxr/assets/5181065/75cd2498-a2c6-4f40-9a38-356b465e01ad)
### CXR-to-report 
![report_gen](https://github.com/hyn2028/llm-cxr/assets/5181065/a3c62d64-9f63-4580-a779-b4bc2f560aa0)
### CXR VQA
![cxr_vqa](https://github.com/hyn2028/llm-cxr/assets/5181065/4a572607-c20b-4634-9443-b20c89983d7c)


## Install dependencies
We use `conda` to manage the environment. Create a new environment using the `environment.yaml` file.

```bash
conda env create --file environment.yaml
```

To activate the conda environment, run the command below.
```bash
conda activate llm-cxr
```

## Pretrained models
We provide checkpoints used in the paper to generate and evaluate results. Using this checkpoint, you can interactively experience and reproduce LLM-CXR in the Gradio environment without training. See the paper for details regarding training of uploaded checkpoints.

| Model | Link |
| :---: |:---: |
| LLM | [link](https://drive.google.com/file/d/1N-0tUNUdy4DaOLKnJXblB8F5G3TqVVBD/view?usp=sharing) |
| VQ-GAN | [link](https://drive.google.com/file/d/1Kh684zK6vKT7Jgx-tBck23_e7siX4HAY/view?usp=sharing) |

### Install pretrained models
#### LLM
Unzip the downloaded `llmcxr_mimic-cxr-256-txvloss-medvqa-stage1_2.tar` file. Place the unzipped `llmcxr_mimic-cxr-256-txvloss-medvqa-stage1_2` directory in the `ckpts/` directory.
#### VQ-GAN
Unzip the downloaded `vqgan_mimic-cxr-256-txvloss.tar` file. Place the unzipped `vqgan_mimic-cxr-256-txvloss` directory in the `ckpts/` directory.

## Interactive demo in local environment using Gradio

![gradio_demo](https://github.com/hyn2028/llm-cxr/assets/5181065/f84e2c92-7eed-4eb0-af12-fea4cb24cc40)


Run the shell script below to run the interactive demo server of LLM-CXR. We recommend using a GPU with at least 11GB of memory such as `NVIDIA GeForce GTX 1080 Ti 11GB` or higher.


```bash
python generate_interactive.py <model_path> <vqgan_config_path> <vqgan_ckpt_path>
```

```bash
python generate_interactive.py \
 ckpts/llmcxr_mimic-cxr-256-txvloss-medvqa-stage1_2 \
 ckpts/vqgan_mimic-cxr-256-txvloss/2023-09-05T13-56-50_mimic-cxr-256-txvloss-project-compat.yaml \
 ckpts/vqgan_mimic-cxr-256-txvloss/2023-09-05T13-56-50_mimic-cxr-256-txvloss-4e-compat.ckpt
```

You can access the demo server at `http://localhost:7860/` in your browser.


## Train LLM-CXR from scratch

### Install additional dependencies
To reproduce results, separate environments must be used for LLM training and inference, and for image encoding and decoding. Install `llm-cxr` and `llm-cxr-taming` conda virtual environment using the script below.
```bash
conda env create --file environment.yaml        # install llm-cxr environment
conda env create --file environment_taming.yaml # install llm-cxr-taming environment
```

### Prepare data

> NOTE: to access the `MIMIC-CXR` dataset family, appropriate credentials are required.

1. MIMIC-CXR dataset: Download the entire `MIMIC-CXR-JPG` dataset from the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset. All downloaded files must be located under `data/mimic-cxr-jpg`. Unzip the metadata files in the `.csv.gz` format at the root of the `data/mimic-cxr-jpg` directory. Then download `mimic-cxr-reports.zip` file from the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) dataset, unzip it, and place the `files/` directory into the `data/mimic-cxr-jpg/reports` directory. 

1. Instruction following dataset: Download `databricks-dolly-15k.jsonl` file from [here](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/orig_dataset) and put it in `data/` directory.

Your final directory structure should look like this:
```
data/
├── mimic-cxr-jpg/
│   ├── files/
│   │   ├── p10/
│   │   ├── ...
│   │   └── p19/
│   ├── reports/
│   │   └── files/
│   │       ├── p10/
│   │       ├── ...
│   │       └── p19/
│   ├── mimic-cxr-jpg_medvqa_v1/
│   │   ├── p10/
│   │   ├── ...
│   │   └── p19/
│   ├── mimic-cxr-2.0.0-metadata
│   ├── mimic-cxr-2.0.0-split
│   ├── mimic-cxr-2.0.0-selected-pa-ap-earlist-study.pickle
│   └── mimic-cxr-2.0.0-selected-pa-ap.pickle
├── databricks-dolly-15k.jsonl
├── eval_dicom_ids.pickle
├── mimic_cxr_img_list_train.txt
├── mimic_cxr_img_list_test.txt
└── mimic-cxr-256-txvloss_codebook_indices.pickle  # see below to generate this file
```
### Training VQ-GAN
Unfortunately, the dependencies of the VQ-GAN library are too old and are not compatible with the LLM-CXR environment. Therefore, the code to train VQ-GAN is not ready now. Instead, use the checkpoints of VQ-GAN that we trained in advance.

### Encode entire CXR images
To encode the entire MIMIC-CXR images with VQ-GAN, run the shell script below. This will create a `mimic-cxr-256-txvloss_codebook_indices.pickle` file in the `data/` directory. This file contains the encoded (vector quantized) entire CXR images. 

```bash
conda activate llm-cxr-taming
python encode_cxr_all.py <vqgan_config_path> <vqgan_ckpt_path> <path_result> <paths_data_list1> <paths_data_list2> ...
```

```bash
conda activate llm-cxr-taming
python encode_cxr_all.py \
 ckpts/vqgan_mimic-cxr-256-txvloss/2023-09-05T13-56-50_mimic-cxr-256-txvloss-project-compat.yaml \
 ckpts/vqgan_mimic-cxr-256-txvloss/2023-09-05T13-56-50_mimic-cxr-256-txvloss-4e-compat.ckpt \
 data/mimic-cxr-256-txvloss_codebook_indices.pickle \
 data/mimic_cxr_img_list_train.txt \
 data/mimic_cxr_img_list_test.txt
```

### Train Model
#### Stage 1
Run the shell script below for the stage 1 train.

```bash
conda activate llm-cxr
./train_llmcxr_stage1.sh
```
#### Stage 2
Run the shell script below for the stage 2 train. 
```bash
conda activate llm-cxr
./tarin_llmcxr_stage2.sh
```
Before running, modify the environment variable `input_model` in the `train_llmcxr_stage2.sh` file to point to the checkpoint path of the model trained in stage1.

#### Convert checkpoint 
The checkpoint of the saved LLM is a ` DeepSpeed zero` checkpoint, thus, must be converted to the `pytorch_model.bin` file for inference or to continue training. Convert the checkpoint using the `zero_to_fp32.py` file created together in the created checkpoint directory. You can simply convert using the script below.

```bash
conda activate llm-cxr
python zero_to_fp32.py . pytorch_model.bin
```

### Training Tips
The current settings are geared towards `NVIDIA A100 40GB x8`, but if you change the `DeepSpeed` settings, you can also train on smaller GPUs like `NVIDIA GeForceRTX 3090 24GB x2`. Please refer to the original [code base repository](https://github.com/databrickslabs/dolly) and change the `DeepSpeed` settings (`config/ds_z3_bf16_config.json`) accordingly. Also adjust the `--num_gpus` argument in the `train_llmcxr_stage*.sh` file to match the number of GPUs.


### Inference for evaluation (Report-to-CXR and CXR-to-Report)
To generate inference results for evaluation, run the shell script below. This will create a `eval_inference/` directory in the root directory. This directory contains the inference results `dolly__eval_results_0_1.pickle`. This file contains the inference results of the report-to-CXR and CXR-to-report tasks from our evaluation dataset `data/eval_dicom_ids.pickle`.

```bash 
conda activate llm-cxr
python generate_eval.py <model_path> <cxr_vq_path> <output_root> 
```
```bash
conda activate llm-cxr
python generate_eval.py ckpts/llmcxr_mimic-cxr-256-txvloss-medvqa-stage1_2 data/mimic-cxr-256-txvloss_codebook_indices.pickle eval_inference
```

### Decode inference results
To decode the inference results, run the shell script below. This will create a `eval_inference_decoded/` directory in the root directory. The `generated_imgs_jpg/` directory contains images generated from reports, and the `generated_reports.txt` file contains reports generated from images. GT reports and generated reports are interleaved in order.

```bash
conda activate llm-cxr-taming
python decode_cxr_all.py <vqgan_config_path> <vqgan_ckpt_path> <output_root> <infer_result_path>
```

```bash
conda activate llm-cxr-taming
python decode_cxr_all.py \
 ckpts/vqgan_mimic-cxr-256-txvloss/2023-09-05T13-56-50_mimic-cxr-256-txvloss-project-compat.yaml \
 ckpts/vqgan_mimic-cxr-256-txvloss/2023-09-05T13-56-50_mimic-cxr-256-txvloss-4e-compat.ckpt \
 eval_inference_decoded \
 eval_inference/llm-cxr__eval_results_0_1.pickle
```


## Acknowledgements
We thank the authors for their great work. 
- We were heavily inspired by [UniXGen](https://arxiv.org/abs/2302.12172) for how we encode images to create them bidirectionally with transformers.
- Our training pipeline was modified from the [Databrick's Dolly](https://github.com/databrickslabs/dolly). 
- We also thank [Taming Transformers](https://github.com/CompVis/taming-transformers) for providing the architecture of `VQ-GAN` as image encoder and decoder. 
