import os
import cv2
import math
import numpy as np
import matplotlib
from PIL import Image
from matplotlib import pyplot as plt

import timm
from timm.data.transforms_factory import create_transform

from transformers import AutoModelForImageClassification

import torch
from torchvision import transforms

import torchvision.models as models

from tqdm import tqdm
import json
from utils import *

results_save_root = "./explanation_insertion_results"
explanation_method = "explanation_results/imagenet-mambavision-true/Rise"
image_root_path = "datasets/imagenet/ILSVRC2012_img_val"
eval_list = "datasets/imagenet/val_mambavision_5k_true.txt"
save_doc = "imagenet-true-mambavision"
steps = 50
batch_size = 10
image_size_ = 224

data_transform = create_transform(input_size=(3, 224, 224),
                             is_training=False,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

def preprocess_input(images):
    """
    Input:
        image: An image read by opencv [b,w,h,c]
    Output:
        outputs: After preproccessing, is a tensor [c,w,h]
    """
    outputs = []
    for image in images:
        image = Image.fromarray(image)
        image = data_transform(image)
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
    return perturbed_image.astype(np.uint8)

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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    mkdir(results_save_root)
    save_dir = os.path.join(results_save_root, save_doc)
    mkdir(save_dir)
    save_dir = os.path.join(save_dir, explanation_method.split("/")[-1])
    mkdir(save_dir)

    model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-L2-1K", trust_remote_code=True)
    model.eval()
    model.to(device)
    model = MambaVision_Super(model, device)
    model.eval()
    model.to(device)

    with open(eval_list, "r") as f:
        infos = f.read().split('\n')

    for info in tqdm(infos[:]):
        json_file = {}
        class_index = int(info.split(" ")[-1])
        image_path = os.path.join(image_root_path, info.split(" ")[0])

        mask_path = os.path.join(explanation_method, info.split(" ")[0].replace(".JPEG", ".npy"))

        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size_, image_size_))
        explanation = np.load(mask_path)

        insertion_explanation_images = []
        deletion_explanation_images = []
        for i in range(1, steps+1):
            perturbed_rate = i / steps
            insertion_explanation_images.append(perturbed(image, explanation, rate = perturbed_rate, mode = "insertion"))
            deletion_explanation_images.append(perturbed(image, explanation, rate = perturbed_rate, mode = "deletion"))
        
        insertion_explanation_images_input = preprocess_input(
            np.array(insertion_explanation_images)
        ).to(device)
        deletion_explanation_images_input = preprocess_input(
            np.array(deletion_explanation_images)
        ).to(device)

        batch_step = math.ceil(
            insertion_explanation_images_input.shape[0] / batch_size)
        
        insertion_data = []
        deletion_data = []
        for j in range(batch_step):
            insertion_explanation_images_input_results = torch.softmax(model(
                insertion_explanation_images_input[j*batch_size:j*batch_size+batch_size])
                                                                       , -1)[:,class_index]
            insertion_data += insertion_explanation_images_input_results.detach().cpu().numpy().tolist()
            
            deletion_explanation_images_input_results = torch.softmax(model(
                deletion_explanation_images_input[j*batch_size:j*batch_size+batch_size])
                                                                      , -1)[:,class_index]
            deletion_data += deletion_explanation_images_input_results.detach().cpu().numpy().tolist()
        
        json_file["consistency_score"] = insertion_data
        json_file["collaboration_score"] = deletion_data
        
        save_path = os.path.join(
            save_dir, info.split(" ")[0].replace(".JPEG", ".json")
        )
        with open(save_path, "w") as f:
            f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
        
    return

main()