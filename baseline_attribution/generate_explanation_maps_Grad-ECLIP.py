"""
Created on 2024/8/19

@author: Ruoyu Chen
"""
import os
from torch.autograd import Variable
from .Grad_Eclip.grad_eclip import *

import numpy as np
import cv2
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torchvision import transforms

# import tensorflow as tf
from utils import *

import clip

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

img_size = 224

# dataset_index = "datasets/imagenet/val_clip_vitl_5k_true.txt"
# SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-clip-vitl-true")

dataset_index = "datasets/imagenet/val_clip_vitl_2k_false.txt"
SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-clip-vitl-false")
    
dataset_path = "datasets/imagenet/ILSVRC2012_img_val"
class_number = 1000
batch = 1
mkdir(SAVE_PATH)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model
    clipmodel, preprocess = clip.load("ViT-L/14", device=device, download_root=".checkpoints/CLIP")
    
    semantic_path = "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
    if os.path.exists(semantic_path):
        text_embedding = torch.load(semantic_path, map_location="cpu")
        text_embedding = text_embedding.to(device)
    
    explainer = Grad_ECLIP(clipmodel, text_embedding)
    
    # data preproccess
    with open(dataset_index, "r") as f:
        datas = f.read().split('\n')
    
    input_data = []
    label = []
    for data in datas:
        label.append(int(data.strip().split(" ")[-1]))
        input_data.append(
            os.path.join(dataset_path, data.split(" ")[0])
        )
    
    total_steps = math.ceil(len(input_data) / batch)
    
    explainer_method_name = "GradECLIP"
    exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
    mkdir(exp_save_path)
    
    for step in tqdm(range(total_steps), desc=explainer_method_name):
        img_path = input_data[step]
        X_raw = Image.open(img_path).convert("RGB").resize((224,224))
        X_raw = imgprocess(X_raw).to(device).unsqueeze(0)

        Y_true = label[step]
        
        torch.cuda.empty_cache() 
        explanation = explainer(
            X_raw,
            Y_true
            )
        torch.cuda.empty_cache() 
        
        np.save(os.path.join(exp_save_path, img_path.split("/")[-1].replace(".JPEG", "")), 
                cv2.resize(explanation.astype(np.float32), (224,224)))

main()