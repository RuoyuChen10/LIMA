# -*- coding: utf-8 -*-  

"""
Created on 2024/8/16

@author: Ruoyu Chen
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import cv2
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import timm
from timm.data.transforms_factory import create_transform

from transformers import AutoModelForImageClassification

from xplique.wrappers import TorchWrapper
from xplique.plots import plot_attributions
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod, HsicAttributionMethod)
import torchvision.models as models

import torch
from torchvision import transforms

import tensorflow as tf
from utils import *

tf.config.run_functions_eagerly(True)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048*4)]
)

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "imagenet"
net_mode  = "vision_mamba" # "resnet", vgg
print(net_mode)
if mode == "imagenet":
    if net_mode == "vitl":
        img_size = 224
        dataset_index = "datasets/imagenet/val_vitl_5k_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-vitl-true")
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
    elif net_mode == "mambavision":
        img_size = 224
        dataset_index = "datasets/imagenet/val_mambavision_5k_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-mambavision-true")
        data_transform = create_transform(input_size=(3, 224, 224),
                             is_training=False,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    elif net_mode == "vision_mamba":
        from vim.models_mamba import vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2
        from timm.models import create_model
        img_size = 224
        dataset_index = "datasets/imagenet/val_vim_5k_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-vim-true")
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
        
    dataset_path = "datasets/imagenet/ILSVRC2012_img_val"
    class_number = 1000
    batch = 100
    mkdir(SAVE_PATH)

class MambaVision_Super(torch.nn.Module):
    def __init__(self, 
                 model,
                 device = "cuda"):
        super().__init__()
        self.model = model
        self.device = device
    
    def forward(self, vision_inputs):
        # with torch.no_grad():
        predicted_scores = self.model(vision_inputs)['logits']
        return predicted_scores

def load_and_transform_vision_data(image_paths, device, channel_first=False):
    if image_paths is None:
        return None

    image_outputs = []
    
    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_outputs.append(image)
    image_outputs = torch.stack(image_outputs, dim=0)
    if channel_first:
        pass
    else:
        image_outputs = image_outputs.permute(0,2,3,1)
    return image_outputs.cpu().numpy()   

def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # Load model
    if net_mode == "vitl":
        vis_model = timm.create_model('vit_large_patch16_224', pretrained=False)
        vis_model.load_state_dict(torch.load('ckpt/pytorch_model/vit_large_patch16_224_pretrained.pth'))  # 加载本地权重
    elif net_mode == "mambavision":
        vis_model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-L2-1K", trust_remote_code=True)
        vis_model = MambaVision_Super(vis_model, device)
    elif net_mode == "vision_mamba":
        vis_model = create_model(
            "vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2",
            pretrained=False,
            num_classes=1000,
            drop_rate=0.01,
            drop_path_rate=0.05,
            drop_block_rate=None,
            img_size=224
        )
        checkpoint = torch.load("ckpt/pytorch_model/vim_b_midclstok_81p9acc.pth", map_location='cpu')

        checkpoint_model = checkpoint['model']
        
        state_dict = vis_model.state_dict()
        
        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = vis_model.patch_embed.num_patches
        num_extra_tokens = vis_model.pos_embed.shape[-2] - num_patches
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

        vis_model.load_state_dict(checkpoint_model, strict=False)
        print("load Vision Mamba model")
        
    vis_model.eval()
    vis_model.to(device)
    
    wrapped_model = TorchWrapper(vis_model.eval(), device)
    
    batch_size = 64
    
    # define explainers
    explainers = [
        Saliency(wrapped_model),
        # GradientInput(model),
        # GuidedBackprop(model),
        IntegratedGradients(wrapped_model, steps=80, batch_size=32),
        # SmoothGrad(model, nb_samples=80, batch_size=batch_size),
        # SquareGrad(model, nb_samples=80, batch_size=batch_size),
        # VarGrad(model, nb_samples=80, batch_size=batch_size),
        # GradCAM(model),
        # GradCAMPP(model),
        # Occlusion(wrapped_model, patch_size=10, patch_stride=5, batch_size=batch_size),
        # SobolAttributionMethod(model, batch_size=batch_size),
        HsicAttributionMethod(wrapped_model, batch_size=batch_size, grid_size=7),
        Rise(wrapped_model, nb_samples=500, batch_size=batch_size),
        # Lime(model, nb_samples = 1000),
        KernelShap(wrapped_model, nb_samples = 1000, batch_size=32)
    ]
    
    # data preproccess
    with open(dataset_index, "r") as f:
        datas = f.read().split('\n')
    
    input_data = []
    label = []
    for data in datas:
        if "34333" in os.path.join(dataset_path, data.split(" ")[0]):
            label.append(int(data.strip().split(" ")[-1]))
            input_data.append(
                os.path.join(dataset_path, data.split(" ")[0])
            )
    
    total_steps = math.ceil(len(input_data) / batch)
    
    for explainer in explainers:
        # explanation methods    
        explainer_method_name = explainer.__class__.__name__
        exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
        mkdir(exp_save_path)
        
        for step in tqdm(range(total_steps), desc=explainer_method_name):
            image_names = input_data[step * batch : step * batch + batch]
            X_raw = load_and_transform_vision_data(image_names, device)

            Y_true = np.array(label[step * batch : step * batch + batch])
            labels_ohe = np.eye(class_number)[Y_true]
            
            explanations = explainer(X_raw, labels_ohe)
            if type(explanations) != np.ndarray:
                explanations = explanations.numpy()
            
            for explanation, image_name in zip(explanations, image_names):
                mkdir(exp_save_path)
                np.save(os.path.join(exp_save_path, image_name.split("/")[-1].replace(".JPEG", "")), explanation)
    
    return

main()