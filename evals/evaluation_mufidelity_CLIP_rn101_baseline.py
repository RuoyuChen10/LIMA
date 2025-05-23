import argparse

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from xplique.wrappers import TorchWrapper
from xplique.metrics import MuFidelity

import clip

import torch
import torchvision.transforms as transforms

from tqdm import tqdm

from utils import *

# tf.config.run_functions_eagerly(True)

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4048)]
# )

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

def transform_vision_data(image_path, channel_first=False):
    """
    Input:
        image: An image read by opencv [w,h,c]
    Output:
        image: After preproccessing, is a tensor [c,w,h]
    """
    image =cv2.imread(image_path)
    
    image = Image.fromarray(image)
    image = data_transform(image)
    if channel_first:
        pass
    else:
        image = image.permute(1,2,0)
    return image.numpy()

def parse_args():
    parser = argparse.ArgumentParser(description='Deletion Metric')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/imagenet/ILSVRC2012_img_val',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/imagenet/val_clip_rn101_5k_true.txt',
                        help='Datasets.')
    parser.add_argument('--eval-number',
                        type=int,
                        default=495,
                        help='Datasets.')
    parser.add_argument('--explanation-method', 
                        type=str, 
                        default='./explanation_results/imagenet-clip-rn101-true/HsicAttributionMethod-8x8',
                        help='Save path for saliency maps generated by interpretability methods.')
    args = parser.parse_args()
    return args

class CLIPModel_Super(torch.nn.Module):
    def __init__(self, 
                 type="ViT-L/14", 
                 download_root=None,
                 device = "cuda"):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(type, device=self.device, download_root=download_root)
        
    def equip_semantic_modal(self, modal_list):
        text = clip.tokenize(modal_list).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.model.encode_text(text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            
    def forward(self, vision_inputs):
        
        with torch.no_grad():
            image_features = self.model.encode_image(vision_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        scores = (image_features @ self.text_features.T).softmax(dim=-1)
        return scores.float()

def main(args):
    class_number = 1000
    batch = 64
    
    # data preproccess
    with open(args.eval_list, "r") as f:
        datas = f.read().split('\n')

    label = []
    input_image = []
    explanations = []

    for data in tqdm(datas[ : args.eval_number]):
        label.append(int(data.strip().split(" ")[-1]))
        input_image.append(
            transform_vision_data(os.path.join(args.Datasets, data.split(" ")[0]))
        )
        explanations.append(
            norm(np.load(
                os.path.join(args.explanation_method, data.split(" ")[0].replace(".JPEG", ".npy"))))
        )
        
    label_onehot = tf.one_hot(np.array(label), class_number)
    input_image = np.array(input_image)
    explanations = np.array(explanations)
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    vis_model = CLIPModel_Super("RN101", download_root=".checkpoints/CLIP")
    vis_model.eval()
    vis_model.to(device)
    
    semantic_path = "ckpt/semantic_features/clip_rn101_imagenet_zeroweights.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)

    vis_model.text_features = semantic_feature
    
    model = TorchWrapper(vis_model.eval(), device)
    torch.cuda.empty_cache()

    # original
    metric = MuFidelity(model, input_image, label_onehot, batch_size=128, grid_size=14)

    mufidelity_score_org = metric(explanations.astype(np.float32))
    
    print("{} Attribution Method MuFidelity Score: {}".format(args.explanation_method.split("/")[-1], mufidelity_score_org))
    return 


if __name__ == "__main__":
    args = parse_args()
    main(args)