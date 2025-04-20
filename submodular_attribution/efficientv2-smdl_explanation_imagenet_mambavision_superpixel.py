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
import imageio
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

from transformers import AutoModelForImageClassification
from PIL import Image
from timm.data.transforms_factory import create_transform
import torch
from torchvision import transforms

red_tr = get_alpha_cmap('Reds')

from models.submodular_single_modal import BlackBoxSingleModalSubModularExplanationEfficient

data_transform = create_transform(input_size=(3, 224, 224),
                             is_training=False,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for ImageBind Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/imagenet/ILSVRC2012_img_val',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/imagenet/val_mambavision_5k_true.txt',
                        help='Datasets.')
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="slico",
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
                        type=str, default='./submodular_results/imagenet-mambavision-efficient/',
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

class MambaVision_Super(torch.nn.Module):
    def __init__(self, 
                 model,
                 device = "cuda"):
        super().__init__()
        self.model = model
        self.device = device
    
    def forward(self, vision_inputs):
        with torch.no_grad():
            predicted_scores = self.model(vision_inputs)['logits']
        return predicted_scores

def main(args):
    # Model Init
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # Instantiate model
    model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-L2-1K", trust_remote_code=True)
    
    model.eval()
    model.to(device)
    
    vis_model = MambaVision_Super(model, device)
    print("load MambaVision model")
    
    smdl = BlackBoxSingleModalSubModularExplanationEfficient(
        vis_model, transform_vision_data, device=device, 
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