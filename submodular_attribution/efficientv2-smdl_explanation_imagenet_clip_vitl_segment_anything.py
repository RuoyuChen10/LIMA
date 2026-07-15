# -*- coding: utf-8 -*-

"""
Created on 2024/6/3

@author: Ruoyu Chen
CLIP ViT version
"""

import argparse

import scipy
import os
import cv2
import json
import imageio
import numpy as np
from PIL import Image

import subprocess
from scipy.ndimage import gaussian_filter
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('seaborn')

from tqdm import tqdm
from utils import *
import time

import clip

import torch
from torchvision import transforms
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

red_tr = get_alpha_cmap('Reds')

from models.submodular_vit_efficient import MultiModalSubModularExplanationEfficientV2

data_transform = transforms.Compose(
    [
        transforms.Resize(
            (224,224), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
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
                        default='datasets/imagenet/val_clip_vitl_5k_true.txt',
                        help='Datasets.')
    parser.add_argument('--sam-model-type',
                        type=str,
                        default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help='SAM model type.')
    parser.add_argument('--sam-checkpoint',
                        type=str,
                        default='ckpt/pytorch_model/sam_vit_h_4b8939.pth',
                        help='SAM checkpoint path.')
    parser.add_argument('--sam-stability-score-thresh',
                        type=float,
                        default=0.8,
                        help='SAM automatic mask stability score threshold.')
    parser.add_argument('--lambda1', 
                        type=float, default=0.,
                        help='')
    parser.add_argument('--lambda2', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--lambda3', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--lambda4', 
                        type=float, default=10.,
                        help='')
    parser.add_argument('--pending-samples',
                        type=int,
                        default=8,
                        help='')
    parser.add_argument('--pending-samples-rate',
                        type=float,
                        default=None,
                        help='If set, override pending samples per image as ceil(num_sam_regions * rate).')
    parser.add_argument('--begin', 
                        type=int, default=0,
                        help='')
    parser.add_argument('--end', 
                        type=int, default=-1,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/imagenet-clip-vitl-efficientv2/',
                        help='output directory to save results')
    parser.add_argument('--record-counterfactual',
                        action='store_true',
                        help='record the original top-1 failure class and its score trajectory')
    args = parser.parse_args()
    return args

def processing_sam_concepts(sam_masks, image):
    """
    Process SAM masks into non-overlapping concept images.
    """
    if len(sam_masks) == 0:
        return [image]

    mask_sets_V = [mask['segmentation'].astype(np.uint8) for mask in sam_masks]
    num = len(mask_sets_V)

    for i in range(num - 1):
        for j in range(i + 1, num):
            intersection_region = (mask_sets_V[i] + mask_sets_V[j] == 2).astype(np.uint8)
            if intersection_region.sum() == 0:
                continue

            proportion_1 = intersection_region.sum() / max(mask_sets_V[i].sum(), 1)
            proportion_2 = intersection_region.sum() / max(mask_sets_V[j].sum(), 1)
            if proportion_1 > proportion_2:
                mask_sets_V[j] -= intersection_region
            else:
                mask_sets_V[i] -= intersection_region

    element_sets_V = []
    for mask in mask_sets_V:
        if mask.mean() > 0.0005:
            element_sets_V.append(image * mask[:, :, np.newaxis])

    if len(element_sets_V) == 0:
        return [image]

    residual = image - np.array(element_sets_V).sum(0).astype(np.uint8)
    if residual.mean() > 0:
        element_sets_V.append(residual)

    return element_sets_V

class CLIPModel_Super(torch.nn.Module):
    def __init__(self, 
                 type="ViT-L/14", 
                 download_root=None,
                 device = "cuda"):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(type, device=self.device, download_root=download_root)
        
    def forward(self, vision_inputs):
        """
        Input:
            vision_inputs: torch.size([B,C,W,H])
        Output:
            embeddings: a d-dimensional vector torch.size([B,d])
        """
        with torch.no_grad():
            image_features = self.model.encode_image(vision_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features

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

def zeroshot_classifier(model, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = model.model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights).to(device)
    return zeroshot_weights * 100

def main(args):
    # Model Init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Instantiate model
    vis_model = CLIPModel_Super("ViT-L/14", download_root=".checkpoints/CLIP")
    vis_model.eval()
    vis_model.to(device)
    print("load CLIP model")

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        stability_score_thresh=args.sam_stability_score_thresh)
    print("load SAM model")
    
    semantic_path = "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)
    else:
        semantic_feature = zeroshot_classifier(
            vis_model, imagenet_classes, imagenet_templates, device
        )
        os.makedirs(os.path.dirname(semantic_path), exist_ok=True)
        torch.save(semantic_feature, semantic_path)
    
    smdl = MultiModalSubModularExplanationEfficientV2(
        vis_model, semantic_feature, transform_vision_data, device=device, 
        lambda1=args.lambda1, 
        lambda2=args.lambda2, 
        lambda3=args.lambda3, 
        lambda4=args.lambda4,
        pending_samples=args.pending_samples,
        record_counterfactual=args.record_counterfactual,
        class_names=imagenet_classes)
    
    with open(args.eval_list, "r") as f:
        infos = f.read().split('\n')
    
    mkdir(args.save_dir)
    pending_label = args.pending_samples
    if args.pending_samples_rate is not None:
        pending_label = "rate-{}".format(args.pending_samples_rate)
    save_dir = os.path.join(args.save_dir, "SAM-{}-{}-{}-{}-pending-samples-{}".format(args.lambda1, args.lambda2, args.lambda3, args.lambda4, pending_label))
    
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
        if not info.strip():
            continue
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
        
        sam_masks = mask_generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        element_sets_V = processing_sam_concepts(sam_masks, image)
        smdl.k = len(element_sets_V)
        if args.pending_samples_rate is not None:
            smdl.pending_samples = max(1, int(np.ceil(smdl.k * args.pending_samples_rate)))
        else:
            smdl.pending_samples = args.pending_samples

    #     start = time.time()
        submodular_image, submodular_image_set, saved_json_file = smdl(element_sets_V, gt_label)
    #     end = time.time()
    #     # print('程序执行时间: ',end - start)
        
        # Save the final image
        # save_image_root_path = os.path.join(save_dir, "image")
        # mkdir(save_image_root_path)
        # mkdir(os.path.join(save_image_root_path, gt_id))
        # save_image_path = os.path.join(
        #     save_image_root_path, image_relative_path)
        # cv2.imwrite(save_image_path, submodular_image)

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

        # Save GIF
        # save_gif_root_path = os.path.join(save_dir, "gif")
        # mkdir(save_gif_root_path)
        # save_gif_path = os.path.join(save_gif_root_path, gt_id)
        # mkdir(save_gif_path)

        # img_frame = submodular_image_set[0][..., ::-1]
        # frames = []
        # frames.append(img_frame)
        # for fps in range(1, submodular_image_set.shape[0]):
        #     img_frame = img_frame.copy() + submodular_image_set[fps][..., ::-1]
        #     frames.append(img_frame)

        # imageio.mimsave(os.path.join(save_gif_root_path, image_relative_path.replace(".jpg", ".gif")), 
        #                       frames, 'GIF', duration=0.0085)  


if __name__ == "__main__":
    args = parse_args()
    
    main(args)
