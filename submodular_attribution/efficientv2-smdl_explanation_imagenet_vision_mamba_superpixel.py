# -*- coding: utf-8 -*-

"""
Created on 2024/8/16

@author: Ruoyu Chen
Swin Large version
"""

import argparse

import scipy
import os
import cv2
import json
import timm
from timm.models import create_model
import numpy as np
from PIL import Image

import subprocess
from scipy.ndimage import gaussian_filter
import matplotlib
from matplotlib import pyplot as plt
# plt.style.use('seaborn')

from tqdm import tqdm
from utils import *
import time

from vim.models_mamba import vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2
import torch
from torchvision import transforms

import torchvision.models as models

red_tr = get_alpha_cmap('Reds')

from models.submodular_single_modal import BlackBoxSingleModalSubModularExplanationEfficient

data_transform = transforms.Compose(
    [
        transforms.Resize(
            (224,224), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for ImageBind Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/imagenet/ILSVRC2012_img_val',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/imagenet/val_vim_5k_true.txt',
                        help='Datasets.')
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="seeds",
                        choices=["slico", "seeds"],
                        help="")
    parser.add_argument('--lambda1', 
                        type=float, default=20.,
                        help='')
    parser.add_argument('--lambda2', 
                        type=float, default=5.,
                        help='')
    parser.add_argument('--lambda3', 
                        type=float, default=0.01,
                        help='')
    parser.add_argument('--pending-samples',
                        type=int,
                        default=12,
                        help='')
    parser.add_argument('--begin', 
                        type=int, default=0,
                        help='')
    parser.add_argument('--end', 
                        type=int, default=None,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/imagenet-vision-mamba-efficient/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def transform_vision_data(image):
    """
    Input:
        image: An image read by opencv [w,h,c]
    Output:
        image: After preproccessing, is a tensor [c,w,h]
    """
    image = Image.fromarray(image)
    image = data_transform(image)
    return image

def main(args):
    # Model Init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Instantiate model
    model = create_model(
        "vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2",
        pretrained=False,
        num_classes=1000,
        drop_rate=0.01,
        drop_path_rate=0.05,
        drop_block_rate=None,
        img_size=224
    )
    # print(model)
    checkpoint = torch.load("ckpt/pytorch_model/vim_b_midclstok_81p9acc.pth", map_location='cpu')

    checkpoint_model = checkpoint['model']
    # print(checkpoint)

    state_dict = model.state_dict()

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)
    model.eval()
    model.to(device)
    print("load Vision Mamba model")
    
    smdl = BlackBoxSingleModalSubModularExplanationEfficient(
        model, transform_vision_data, device=device, 
        lambda1=args.lambda1, 
        lambda2=args.lambda2, 
        lambda3=args.lambda3, 
        pending_samples=args.pending_samples)
    
    with open(args.eval_list, "r") as f:
        infos = f.read().split('\n')
    
    mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, "{}-{}-{}-{}-pending-samples-{}".format(args.superpixel_algorithm, args.lambda1, args.lambda2, args.lambda3, args.pending_samples))  
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_json_root_path = os.path.join(save_dir, "json")
    mkdir(save_json_root_path)
    
    end = args.end
    if end == -1:
        end = None
    select_infos = infos[args.begin : end]
    for info in tqdm(select_infos):
        gt_id = info.split(" ")[1]
        
        image_relative_path = info.split(" ")[0]
        
        if os.path.exists(
            os.path.join(
            os.path.join(save_json_root_path, gt_id), image_relative_path.replace(".JPEG", ".json"))
        ):
            continue
        
        # Ground Truth Label
        gt_label = int(gt_id)
        
        # Read original image
        image_path = os.path.join(args.Datasets, image_relative_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        
        element_sets_V = SubRegionDivision(image, mode=args.superpixel_algorithm)
        smdl.k = len(element_sets_V)

        submodular_image, submodular_image_set, saved_json_file = smdl(element_sets_V, gt_label)

        # Save npy file
        mkdir(os.path.join(save_npy_root_path, gt_id))
        np.save(
            os.path.join(
                os.path.join(save_npy_root_path, gt_id), image_relative_path.replace(".JPEG", ".npy")),
            np.array(submodular_image_set)
        )

        # Save json file
        mkdir(os.path.join(save_json_root_path, gt_id))
        with open(os.path.join(
            os.path.join(save_json_root_path, gt_id), image_relative_path.replace(".JPEG", ".json")), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))

if __name__ == "__main__":
    args = parse_args()
    
    main(args)