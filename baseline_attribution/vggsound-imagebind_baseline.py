# -*- coding: utf-8 -*- 


"""
Created on 2024/8/22

@author: Ruoyu Chen
"""

import os

import numpy as np
import cv2
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from imagebind import data
from imagebind.data import waveform2melspec, get_clip_timepoints
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from xplique.wrappers import TorchWrapper
from xplique.plots import plot_attributions
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod, HsicAttributionMethod)

import torch
from torchvision import transforms
from utils import *

import torchaudio
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

clip_sampler = ConstantClipsPerVideoSampler(
    clip_duration=2, clips_per_video=3
)
# import tensorflow as tf

# tf.config.run_functions_eagerly(True)

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048*5)]
# )

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "vggsound"
net_mode  = "imagebind" # "resnet", vgg

if mode == "vggsound":
    if net_mode == "imagebind":
        img_size = 224
        dataset_index = "datasets/vggsound/val_imagebind_600_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "vggsound-imagebind-true")
    # elif net_mode == "languagebind":
        
    dataset_path = "datasets/vggsound/test"
    class_number = 309
    batch = 16
    mkdir(SAVE_PATH)
    
class ImageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.semantic_modal = None
        
    def forward(self, audio_inputs):
        """
        Input:
            audio_inputs: torch.size([B,C,W,H])
        Output:
            embeddings: a d-dimensional vector torch.size([B,d])
        """
        audio_inputs = audio_inputs.unsqueeze(2)
        
        inputs = {
            "audio": audio_inputs,
        }
        
        # with torch.no_grad():
        embeddings = self.base_model(inputs)
        
        scores = torch.softmax(embeddings["audio"] @ self.semantic_modal.T, dim=-1)
        return scores
    
def read_audio(
    audio_path,
    device="cpu",
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    mean= -4.268, 
    std= 9.138
):
    waveform, sr = torchaudio.load(audio_path)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=sample_rate
        )
    all_clips_timepoints = get_clip_timepoints(
        clip_sampler, waveform.size(1) / sample_rate
    )
    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        waveform_clip = waveform[
            :,
            int(clip_timepoints[0] * sample_rate) : int(
                clip_timepoints[1] * sample_rate
            ),
        ]
        waveform_melspec = waveform2melspec(
            waveform_clip, sample_rate, num_mel_bins, target_length
        )
        all_clips.append(waveform_melspec)

    normalize = transforms.Normalize(mean=mean, std=std)
    all_clips = [normalize(ac).to(device) for ac in all_clips]

    all_clips = torch.stack(all_clips, dim=0)
    
    out = all_clips[:,0,:,:].cpu().numpy()
    out = out.transpose(1,2,0)
    return out

def load_and_transform_audio_data(audio_paths):
    data = []
    for audio_path in audio_paths:
        data.append(read_audio(audio_path))
        
    return np.array(data)

def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # Load model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    
    audio_model = ImageBindModel_Super(model)
    print("load imagebind model")
    
    semantic_path = "ckpt/semantic_features/vggsound_imagebind_cls.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device) * 0.05

    audio_model.semantic_modal = semantic_feature
    
    wrapped_model = TorchWrapper(audio_model.eval(), device)
    
    batch_size = 16
    
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
        # Occlusion(wrapped_model, patch_size=10, patch_stride=5, batch_size=batch_size),
        # Rise(wrapped_model, nb_samples=500, batch_size=batch_size),
        # SobolAttributionMethod(wrapped_model, batch_size=batch_size),
        # HsicAttributionMethod(wrapped_model, batch_size=batch_size),
        # Rise(wrapped_model, nb_samples=500, batch_size=batch_size),
        Lime(wrapped_model, nb_samples = 1000, batch_size=batch_size),
        KernelShap(wrapped_model, nb_samples = 1000, batch_size=batch_size)
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
            X_raw = load_and_transform_audio_data(image_names)

            Y_true = np.array(label[step * batch : step * batch + batch])
            labels_ohe = np.eye(class_number)[Y_true]
            
            print(X_raw.shape)
            explanations = explainer(X_raw, labels_ohe)
            if type(explanations) != np.ndarray:
                explanations = explanations.numpy()
            
            for explanation, image_name in zip(explanations, image_names):
                mkdir(exp_save_path)
                np.save(os.path.join(exp_save_path, image_name.split("/")[-1].replace(".flac", "")), explanation)
    
    return

main()