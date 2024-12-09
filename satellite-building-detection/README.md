# Satellite Building Detection

A semantic segmentation solution for detecting buildings in low-resolution satellite imagery.

## Overview

This project implements a building detection system using 10m resolution satellite imagery. The solution uses deep learning-based semantic segmentation to identify building footprints from three-channel (BGR) satellite data.

## Technical Details

### Dataset

- Input: 256x256 pixel satellite images (2.56km x 2.56km area)
- Resolution: 10m per pixel
- Channels: Blue-Green-Red (raw sensor readings)
- Labels: Binary masks (1: buildings, 0: background)
- Format: Python pickle file containing training and validation sets

## Setup Instructions

### Directory Structure

```bash
mkdir pretrain_model
mkdir data
```

### Environment Setup

```bash
micromamba create -f mamba-env.yaml
```

### Data Preparation

1. Place `dataset.pickle` in `data/`
2. Download `pvt_v2_b0.pth` from [here](https://drive.google.com/drive/folders/1hCaKNrlMF6ut0b36SedPRNC_434R8VVa)
3. Place the downloaded model in `pretrain_model/`

## Usage

### Logging

Modify the configuration in `run.py` to set up your preferred logging solution (default: wandb).

### Model Export

1. Configure `ckpt_path` to point to your desired checkpoint
2. Ensure the current configuration matches training settings

## References

### Citations

| Name                                                                                   | Github                                                                           | Paper                                             |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------- |
| UANet: an Uncertainty-Aware Network for Building Extraction from Remote Sensing Images | [Github](https://github.com/Henryjiepanli/Uncertainty-aware-Network/tree/master) | [Paper](https://arxiv.org/pdf/2307.12309v1)       |
| Pyramid Vision Transformer                                                             | [Github](https://github.com/whai362/PVT)                                         | [Paper](https://arxiv.org/pdf/2106.13797)         |
| High-Resolution Building and Road Detection from Sentinel-2                            | -                                                                                | [Paper](https://arxiv.org/abs/2310.11622)         |
| Benchmark for Building Segmentation on Up-Scaled Sentinel-2 Imagery                    | -                                                                                | [Paper](https://www.mdpi.com/2072-4292/15/9/2347) |

### Related Datasets

- WHU building dataset
- Inria building dataset
- Massachusetts building dataset
