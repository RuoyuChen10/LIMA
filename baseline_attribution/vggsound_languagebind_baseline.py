# -*- coding: utf-8 -*- 


"""
Created on 2024/8/23

@author: Ruoyu Chen
"""
import os

import numpy as np
import cv2
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

from xplique.wrappers import TorchWrapper
from xplique.plots import plot_attributions
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod, HsicAttributionMethod)

import torch
from torchvision import transforms
from utils import *

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "vggsound"
net_mode  = "languagebind" # "resnet", vgg

if mode == "vggsound":
    if net_mode == "languagebind":
        dataset_index = "datasets/vggsound/val_languagebind_600_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "vggsound-languagebind-true")
    # elif net_mode == "languagebind":
        
    dataset_path = "datasets/vggsound/test"
    class_number = 309
    batch = 32
    mkdir(SAVE_PATH)
    
class LanguageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.semantic_modal = None
        
    def forward(self, audio_inputs):
        """
        Input:
            audio_inputs: torch.size([B,C,W,H]) # video
        Output:
            embeddings: a d-dimensional vector torch.size([B,d])
        """
        inputs = {
            "audio": {'pixel_values': audio_inputs},
        }
        
        with torch.no_grad():
            embeddings = self.base_model(inputs)
            
        scores = torch.softmax(embeddings["audio"] @ self.semantic_modal.T, dim=-1)
        
        return scores
    
def read_audio(
    audio_path,
    modality_transform,
    device = "cpu"
):
    audio = [audio_path]
    audio_proccess = to_device(modality_transform['audio'](audio), device)['pixel_values'][0]
    return audio_proccess.cpu().numpy().transpose(1,2,0) # [w,h,c]

def load_and_transform_audio_data(audio_paths, modality_transform):
    data = []
    for audio_path in audio_paths:
        data.append(read_audio(audio_path, modality_transform))
        
    return np.array(data)

def main():
    # Model Init
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # Instantiate model
    clip_type = {
        'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
    }
    model = LanguageBind(clip_type=clip_type, cache_dir='.checkpoints')
    model.eval()
    model.to(device)
    
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}
    
    audio_model = LanguageBindModel_Super(model)
    print("load languagebind model")
    
    semantic_path = "ckpt/semantic_features/vggsound_languagebind_cls.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)
    
    audio_model.semantic_modal = semantic_feature
    
    wrapped_model = TorchWrapper(audio_model.eval(), device)
    
    batch_size = 64
    
    # define explainers
    explainers = [
        # Saliency(wrapped_model),
        # GradientInput(wrapped_model),
        # GuidedBackprop(model),
        # IntegratedGradients(wrapped_model, steps=80, batch_size=batch_size),
        # SmoothGrad(wrapped_model, nb_samples=80, batch_size=batch_size),
        # SquareGrad(wrapped_model, nb_samples=80, batch_size=batch_size),
        # VarGrad(wrapped_model, nb_samples=80, batch_size=batch_size),
        # GradCAM(model),
        # GradCAMPP(model),
        # Occlusion(wrapped_model, patch_size=(10, 45), patch_stride=(5, 20), batch_size=batch_size),
        # Rise(wrapped_model, nb_samples=500, batch_size=batch_size),
        # SobolAttributionMethod(wrapped_model, batch_size=batch_size),
        # HsicAttributionMethod(wrapped_model, batch_size=batch_size),
        # Rise(wrapped_model, nb_samples=500, batch_size=batch_size),
        Lime(wrapped_model, nb_samples = 1000, batch_size=batch_size, distance_mode="cosine"),
        # KernelShap(wrapped_model, nb_samples = 1000, batch_size=batch_size)
    ]
    
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
    
    for explainer in explainers:
        # explanation methods    
        explainer_method_name = explainer.__class__.__name__
        exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
        mkdir(exp_save_path)
        
        for step in tqdm(range(total_steps), desc=explainer_method_name):
            image_names = input_data[step * batch : step * batch + batch]
            X_raw = load_and_transform_audio_data(image_names, modality_transform)

            Y_true = np.array(label[step * batch : step * batch + batch])
            labels_ohe = np.eye(class_number)[Y_true]
            
            explanations = explainer(X_raw, labels_ohe)
            if type(explanations) != np.ndarray:
                explanations = explanations.numpy()
            
            for explanation, image_name in zip(explanations, image_names):
                mkdir(exp_save_path)
                np.save(os.path.join(exp_save_path, image_name.split("/")[-1].replace(".flac", "")), explanation)
    
    return

main()