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

import torch
from torchvision import transforms

# import tensorflow as tf
from utils import *

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "imagenet"
net_mode  = "CLIP" # ImageBind

if mode == "imagenet":
    if net_mode == "CLIP":
        import clip
        img_size = 224
        dataset_index = "datasets/imagenet/val_clip_vitl_5k_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-clip-vitl-true")
    elif net_mode == "ImageBind":
        from imagebind import data
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        img_size = 224
        dataset_index = "datasets/imagenet/val_imagebind_5k_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-imagebind-true")
    elif net_mode == "LanguageBind":
        from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
        img_size = 224
        dataset_index = "datasets/imagenet/val_languagebind_5k_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-languagebind-true")
    elif net_mode == "resnet":
        img_size = 224
        dataset_index = "datasets/imagenet/val_rn101_5k_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-rn101-true")
        
    init(img_size)
    dataset_path = "datasets/imagenet/ILSVRC2012_img_val"
    class_number = 1000
    batch = 16
    mkdir(SAVE_PATH)
    
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
        
        # with torch.no_grad():
        image_features = self.model.encode_image(vision_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        scores = image_features @ self.text_features.T
        return scores.float()
    
class ImageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model, device):
        super().__init__()
        self.base_model = base_model
        self.device = device
        
    def mode_selection(self, mode):
        if mode not in ["text", "audio", "thermal", "depth", "imu"]:
            print("mode {} does not comply with the specification, please select from \"text\", \"audio\", \"thermal\", \"depth\", \"imu\".".format(mode))
        else:
            self.mode = mode
            print("Select mode {}".format(mode))
            
    def equip_semantic_modal(self, modal_list):
        if self.mode == "text":
            self.semantic_modal = data.load_and_transform_text(modal_list, self.device)
        elif self.mode == "audio":
            self.semantic_modal = data.load_and_transform_audio_data(modal_list, self.device)
        
        input = {
                # "vision": vision_inputs,
                self.mode: self.semantic_modal
            }
        with torch.no_grad():
            self.semantic_modal = self.base_model(input)[self.mode]
        print("Equip with {} modal.".format(self.mode))
        
    def forward(self, vision_inputs):
        inputs = {
            "vision": vision_inputs,
        }
        
        # with torch.no_grad():
        embeddings = self.base_model(inputs)
        
        scores = embeddings["vision"] @ self.semantic_modal.T
        return scores

class LanguageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model, device,
                 pretrained_ckpt = f'lb203/LanguageBind_Image',):
        super().__init__()
        self.base_model = base_model
        self.device = device
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(
            pretrained_ckpt, cache_dir='.checkpoints/tokenizer_cache_dir')
        
        self.clip_type = ["video", "audio", "thermal", "image", "depth"]
        self.modality_transform = {c: transform_dict[c](self.base_model.modality_config[c]) for c in self.clip_type}
    
    def mode_selection(self, mode):
        if mode not in ["image", "audio", "video", "depth", "thermal", "language"]:
            print("mode {} does not comply with the specification, please select from \"image\", \"audio\", \"video\", \"depth\", \"thermal\", \"language\".".format(mode))
        else:
            self.mode = mode
            print("Select mode {}".format(mode))
    
    def equip_semantic_modal(self, modal_list):
        if self.mode == "language":
            self.semantic_modal = to_device(self.tokenizer(modal_list, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), self.device)
        elif self.mode in self.clip_type:
            self.semantic_modal = to_device(self.modality_transform[self.mode](modal_list), self.device)
        
        input = {
                # "vision": vision_inputs,
                self.mode: self.semantic_modal
            }
        # with torch.no_grad():
        self.semantic_modal = self.base_model(input)[self.mode]
        print("Equip with {} modal.".format(self.mode))
    
    def forward(self, vision_inputs):
        """
        Input:
            vision_inputs: 
        """
        vision_inputs = vision_inputs.unsqueeze(2)
        vision_inputs = vision_inputs.repeat(1,1,8,1,1)
        inputs = {
            "video": {'pixel_values': vision_inputs},
        }
        
        # with torch.no_grad():
        embeddings = self.base_model(inputs)
            
        scores =embeddings["video"] @ self.semantic_modal.T
        return scores
    
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

def load_and_transform_vision_data(image_paths, device, channel_first=True):
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
    return image_outputs

def load_and_transform_blur_data(image_paths, device):
    if image_paths is None:
        return None

    image_outputs = []
    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB").resize((224,224))
        blurred_image = cv2.GaussianBlur(np.array(image), (51, 51), sigmaX=50)
        blurred_image = Image.fromarray(blurred_image.astype(np.uint8))
        
        blurred_image = data_transform(blurred_image).to(device)
        image_outputs.append(blurred_image)
        
    image_outputs = torch.stack(image_outputs, dim=0)
    return image_outputs

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model
    if net_mode == "CLIP":
        vis_model = CLIPModel_Super("ViT-L/14", download_root=".checkpoints/CLIP")
        vis_model.eval()
        vis_model.to(device)
        
        semantic_path = "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
        if os.path.exists(semantic_path):
            semantic_feature = torch.load(semantic_path, map_location="cpu")
            semantic_feature = semantic_feature.to(device)

        vis_model.text_features = semantic_feature
    
    elif net_mode == "ImageBind":
        vis_model = imagebind_model.imagebind_huge(pretrained=True)
        vis_model.eval()
        vis_model.to(device)
        
        vis_model = ImageBindModel_Super(vis_model, device)
        vis_model.mode_selection("text")
        
        semantic_path = "ckpt/semantic_features/imagebind_imagenet_zeroweights.pt"
        if os.path.exists(semantic_path):
            semantic_feature = torch.load(semantic_path, map_location="cpu")
            semantic_feature = semantic_feature.to(device)
        vis_model.semantic_modal = semantic_feature
        
    elif net_mode == "LanguageBind":
        device = torch.device(device)
        # Load model
        clip_type = {
            'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
            'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
            'thermal': 'LanguageBind_Thermal',
            'image': 'LanguageBind_Image',
            'depth': 'LanguageBind_Depth',
        }
        vis_model = LanguageBind(clip_type=clip_type, cache_dir='.checkpoints')
        vis_model = vis_model.to(device)
        vis_model.eval()
        
        # pretrained_ckpt = f'lb203/LanguageBind_Image'
        # tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='.checkpoints/tokenizer_cache_dir')
        
        semantic_path = "ckpt/semantic_features/languagebind_imagenet_zeroweights.pt"
        if os.path.exists(semantic_path):
            semantic_feature = torch.load(semantic_path, map_location="cpu")
            semantic_feature = semantic_feature.to(device)
            
        vis_model = LanguageBindModel_Super(vis_model, device)
        
        vis_model.semantic_modal = semantic_feature
        print("load languagebind model")
    
    elif net_mode == "resnet":
        import torchvision.models as models
        # Load model
        vis_model = models.resnet101(pretrained = True)
        vis_model.eval()
        vis_model.to(device)
        print("load single-modal resnet-101 model")
     
    # baseline_img = np.zeros((224,224,3))
    # baseline_data = transform_vision_data(baseline_img.astype(np.uint8))
    # baseline_data = baseline_data.unsqueeze(0)
    # print(baseline_data.shape)
    # baseline_data = baseline_data.unsqueeze(0).repeat(batch, 1, 1, 1)
    
    explainer = iGOS_pp
    
    # data preproccess
    with open(dataset_index, "r") as f:
        datas = f.read().split('\n')
    
    input_data = []
    label = []
    for data in datas:
        if "00023943" in data.split(" ")[0]:
            label.append(int(data.strip().split(" ")[-1]))
            input_data.append(
                os.path.join(dataset_path, data.split(" ")[0])
            )
    
    total_steps = math.ceil(len(input_data) / batch)
    
    attr_size=7
    explainer_method_name = "IGOS++-{}x{}".format(attr_size,attr_size)
    exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
    mkdir(exp_save_path)
    
    for step in tqdm(range(total_steps), desc=explainer_method_name):
        image_names = input_data[step * batch : step * batch + batch]
        X_raw = load_and_transform_vision_data(image_names, device)

        Y_true = torch.from_numpy(np.array(label[step * batch : step * batch + batch])).to(device)
        # labels_ohe = np.eye(class_number)[Y_true]
        # num = X_raw.shape[0]
        
        baseline_data = load_and_transform_blur_data(image_names, device)
        
        explanations = explainer(
            vis_model, 
            images=X_raw,
            baselines=baseline_data,
            labels = Y_true,
            size = attr_size
            )
        if type(explanations) != np.ndarray:
            explanations = explanations.detach().cpu().numpy()
        
        for explanation, image_name in zip(explanations, image_names):
            mkdir(exp_save_path)
            np.save(os.path.join(exp_save_path, image_name.split("/")[-1].replace(".JPEG", "")), 
                    cv2.resize(-explanation[0], (224,224)))
            
main()