"""
Created on 2024/8/19

@author: Ruoyu Chen
"""
from torch.autograd import Variable
from .IGOS_pp.methods_helper import *
from .IGOS_pp.method import *


import numpy as np
import cv2
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import clip

import torch
from torchvision import transforms

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.data import waveform2melspec, get_clip_timepoints

# from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
# import tensorflow as tf
from utils import *

import torchaudio
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

clip_sampler = ConstantClipsPerVideoSampler(
    clip_duration=2, clips_per_video=3
)

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "vggsound"
net_mode  = "ImageBind" # ImageBind

if mode == "vggsound":
    if net_mode == "ImageBind":
        # img_size = 224
        dataset_index = "datasets/vggsound/val_imagebind_600_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-imagebind-true")
    elif net_mode == "LanguageBind":
        # img_size = 224
        dataset_index = "datasets/imagenet/val_languagebind_5k_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-languagebind-true")
    
    init((128,204))
    dataset_path = "datasets/vggsound/test"
    class_number = 309
    batch = 1
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
        
        scores = embeddings["audio"] @ self.semantic_modal.T
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
    
    out = all_clips[:,0,:,:]
    # out = out.transpose(1,2,0)
    return out

def load_and_transform_audio_data(audio_paths, device="cuda"):
    data = []
    for audio_path in audio_paths:
        data.append(read_audio(audio_path, device))
        
    return torch.stack(data)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model
    if net_mode == "ImageBind":
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        model.to(device)
        
        audio_model = ImageBindModel_Super(model)
        
        semantic_path = "ckpt/semantic_features/vggsound_imagebind_cls.pt"
        if os.path.exists(semantic_path):
            semantic_feature = torch.load(semantic_path, map_location="cpu")
            semantic_feature = semantic_feature.to(device)
        audio_model.semantic_modal = semantic_feature
        
    elif net_mode == "LanguageBind":
        pass
     
    baseline_data = torch.zeros((128,204,3))
    # baseline_data = transform_vision_data(baseline_img.astype(np.uint8))
    # baseline_data = baseline_data.unsqueeze(0)
    # print(baseline_data.shape)
    baseline_data = baseline_data.unsqueeze(0).repeat(batch, 1, 1, 1)
    
    explainer = iGOS_pp
    
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
    
    explainer_method_name = "IGOS++"
    exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
    mkdir(exp_save_path)
    
    for step in tqdm(range(total_steps), desc=explainer_method_name):
        image_names = input_data[step * batch : step * batch + batch]
        X_raw = load_and_transform_audio_data(image_names, device)

        Y_true = torch.from_numpy(np.array(label[step * batch : step * batch + batch])).to(device)
        # labels_ohe = np.eye(class_number)[Y_true]
        num = X_raw.shape[0]
        
        explanations = explainer(
            audio_model, 
            images=X_raw,
            baselines=baseline_data[:num].to(device),
            labels = Y_true,
            size = 7
            )
        if type(explanations) != np.ndarray:
            explanations = explanations.detach().cpu().numpy()
        
        for explanation, image_name in zip(explanations, image_names):
            mkdir(exp_save_path)
            np.save(os.path.join(exp_save_path, image_name.split("/")[-1].replace(".JPEG", "")), 
                    cv2.resize(-explanation[0], (128,204)))
            
main()