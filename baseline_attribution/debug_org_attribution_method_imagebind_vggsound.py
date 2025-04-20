import os
import cv2
import math
import numpy as np
import matplotlib
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms

from imagebind.data import waveform2melspec, get_clip_timepoints
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from tqdm import tqdm
import json
from utils import *

import torchaudio
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

clip_sampler = ConstantClipsPerVideoSampler(
    clip_duration=2, clips_per_video=3
)

results_save_root = "./explanation_insertion_results"
explanation_method = "explanation_results/vggsound-imagebind-false/HsicAttributionMethod"
image_root_path = "datasets/vggsound/test"
eval_list = "datasets/vggsound/val_imagebind_309_false.txt"
save_doc = "vggsound-imagebind-false"
steps = 50
batch_size = 20

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
        
        with torch.no_grad():
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

def preprocess_input(images):
    """
    Input:
        image: An image read by opencv [b,w,h,c]
    Output:
        outputs: After preproccessing, is a tensor [c,w,h]
    """
    outputs = []
    for image in images:
        image = torch.from_numpy(image)
        image = image.permute(2,0,1).contiguous()
        outputs.append(image)
    return torch.stack(outputs)

def perturbed(image, mask, rate = 0.5, mode = "insertion"):
    mask_flatten = mask.flatten()
    number = int(len(mask_flatten) * rate)
    
    if mode == "insertion":
        new_mask = np.zeros_like(mask_flatten)
        index = np.argsort(-mask_flatten)
        new_mask[index[:number]] = 1

        
    elif mode == "deletion":
        new_mask = np.ones_like(mask_flatten)
        index = np.argsort(mask_flatten)
        new_mask[index[:number]] = 0
    
    new_mask = new_mask.reshape((mask.shape[0], mask.shape[1], 1))
    
    perturbed_image = image * new_mask
    return perturbed_image

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    mkdir(results_save_root)
    save_dir = os.path.join(results_save_root, save_doc)
    mkdir(save_dir)
    save_dir = os.path.join(save_dir, explanation_method.split("/")[-1])
    mkdir(save_dir)

    model_bind = imagebind_model.imagebind_huge(pretrained=True)
    model_bind.eval()
    model_bind.to(device)
        
    model = ImageBindModel_Super(model_bind)
    
    semantic_path = "ckpt/semantic_features/vggsound_imagebind_cls.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)

    model.semantic_modal = semantic_feature

    with open(eval_list, "r") as f:
        infos = f.read().split('\n')

    for info in tqdm(infos):
        json_file = {}
        class_index = int(info.split(" ")[-1])
        audio_path = os.path.join(image_root_path, info.split(" ")[0])

        mask_path = os.path.join(explanation_method, info.split(" ")[0].replace(".flac", ".npy"))

        audio_mfcc = read_audio(audio_path)
        explanation = np.load(mask_path)

        insertion_explanation_audios = []
        deletion_explanation_audios = []
        for i in range(1, steps+1):
            perturbed_rate = i / steps
            insertion_explanation_audios.append(perturbed(audio_mfcc, explanation, rate = perturbed_rate, mode = "insertion"))
            deletion_explanation_audios.append(perturbed(audio_mfcc, explanation, rate = perturbed_rate, mode = "deletion"))
        
        insertion_explanation_audios_input = preprocess_input(
            np.array(insertion_explanation_audios)
        ).to(device)
        deletion_explanation_audios_input = preprocess_input(
            np.array(deletion_explanation_audios)
        ).to(device)

        batch_step = math.ceil(
            insertion_explanation_audios_input.shape[0] / batch_size)
        
        insertion_data = []
        deletion_data = []
        for j in range(batch_step):
            insertion_explanation_audios_input_results = model(
                insertion_explanation_audios_input[j*batch_size:j*batch_size+batch_size])[:,class_index]
            insertion_data += insertion_explanation_audios_input_results.cpu().numpy().tolist()
            
            deletion_explanation_audios_input_results = model(
                deletion_explanation_audios_input[j*batch_size:j*batch_size+batch_size])[:,class_index]
            deletion_data += deletion_explanation_audios_input_results.cpu().numpy().tolist()
        
        json_file["consistency_score"] = insertion_data
        json_file["collaboration_score"] = deletion_data
        
        json_file["org_score"] = insertion_data[-1]
        json_file["baseline_score"] = deletion_data[-1]
        
        save_path = os.path.join(
            save_dir, info.split(" ")[0].replace(".flac", ".json")
        )
        with open(save_path, "w") as f:
            f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
        
    return

main()