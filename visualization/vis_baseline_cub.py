# -*- coding: utf-8 -*-

"""
Created on 2024/8/14

@author: Ruoyu Chen
"""

import argparse

import scipy
import os
import cv2
import json
import imageio
import numpy as np
from PIL import Image

import matplotlib
from matplotlib import pyplot as plt

from tqdm import tqdm
from utils import *

matplotlib.get_cachedir()
plt.rc('font', family="Times New Roman")

from sklearn import metrics

img_size = 224 # 224

def makedirs(dir_save_path):
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Faithfulness Metric')
    parser.add_argument('--image-dir', 
                        type=str, 
                        default='datasets/CUB/test',
                        help='')
    parser.add_argument('--json-dir', 
                        type=str, 
                        default='explanation_insertion_results/cub-fair-mobilenetv2/GradCAMPP',
                        help='')
    parser.add_argument('--npy-dir', 
                        type=str, 
                        default='explanation_results/cub-mobilenetv2/GradCAMPP',
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, 
                        default="./baseline_visualization/cub-mobilenetv2-false/HsicAttributionMethod",
                        help='')
    parser.add_argument('--attr-name', 
                        type=str, 
                        default="HSIC-Attribution",
                        help='')
    args = parser.parse_args()
    return args

def gen_cam(image, mask):
    """
    Generate heatmap
        :param image: [H,W,C]
        :param mask: [H,W],range 0-1
        :return: tuple(cam,heatmap)
    """
    # Read image
    # image = cv2.resize(cv2.imread(image_path), (224,224))
    # mask->heatmap  cv2.COLORMAP_COOL cv2.COLORMAP_JET
    heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_COOL)
    heatmap = np.float32(heatmap)

    # merge heatmap to original image
    cam = 0.5*heatmap + 0.5*np.float32(image)
    return cam, (heatmap).astype(np.uint8)

def norm_image(image):
    """
    Normalization image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

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

def visualization(image, attribution_map, saved_json_file, vis_image, index=None, attr_name="Attribution Map"):
    
    insertion_ours_images = []
    # deletion_ours_images = []

    insertion_image = image - image
    insertion_ours_images.append(insertion_image)
    # deletion_ours_images.append(image - insertion_image)
    for i in range(1, len(saved_json_file["recognition_score"])+1):
        insertion_ours_images.append(
            perturbed(image, attribution_map, rate = i/(len(saved_json_file["recognition_score"])), mode = "insertion"))
    
    insertion_ours_images_input_results = np.array([0.0] + saved_json_file["recognition_score"])

    if index == None:
        ours_best_index = np.argmax(insertion_ours_images_input_results)
    else:
        ours_best_index = index
    x = [i/len(saved_json_file["recognition_score"]) for i in range(0, len(insertion_ours_images_input_results))]
    # i = len(x)

    fig, [ax1, ax2, ax3] = plt.subplots(1,3, gridspec_kw = {'width_ratios':[1, 1, 1.5]}, figsize=(30,8))
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_title(attr_name, fontsize=54)
    ax1.set_facecolor('white')
    ax1.imshow(vis_image[...,::-1].astype(np.uint8))
    
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.xaxis.set_visible(True)
    ax2.yaxis.set_visible(False)
    ax2.set_title('Searched Region', fontsize=54)
    ax2.set_facecolor('white')
    ax2.set_xlabel("Highest conf. {:.4f}".format(insertion_ours_images_input_results.max()), fontsize=44)
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax3.set_xlim((0, 1))
    ax3.set_ylim((0, 1))
    
    ax3.set_ylabel('Recognition Score', fontsize=44)
    ax3.set_xlabel('Percentage of image revealed', fontsize=44)
    ax3.tick_params(axis='both', which='major', labelsize=36)

    x_ = x#[:i]
    ours_y = insertion_ours_images_input_results#[:i]
    ax3.plot(x_, ours_y, color='dodgerblue', linewidth=3.5)  # draw curve
    ax3.set_facecolor('white')
    ax3.spines['bottom'].set_color('black')
    ax3.spines['bottom'].set_linewidth(2.0)
    ax3.spines['top'].set_color('none')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(2.0)
    ax3.spines['right'].set_color('none')

    # plt.legend(["Ours"], fontsize=40, loc="upper left")
    ax3.scatter(x_[-1], ours_y[-1], color='dodgerblue', s=54)  # Plot latest point
    # 在曲线下方填充淡蓝色
    ax3.fill_between(x_, ours_y, color='dodgerblue', alpha=0.1)

    kernel = np.ones((3, 3), dtype=np.uint8)
    # ax3.plot([x_[ours_best_index], x_[ours_best_index]], [0, 1], color='red', linewidth=3.5)  # 绘制红色曲线
    ax3.axvline(x=x_[ours_best_index], color='red', linewidth=3.5)  # 绘制红色垂直线

    # Ours
    mask = (image - insertion_ours_images[ours_best_index]).mean(-1)
    mask[mask>0] = 1

    if ours_best_index != 0:
        dilate = cv2.dilate(mask, kernel, 3)
        # erosion = cv2.erode(dilate, kernel, iterations=3)
        # dilate = cv2.dilate(erosion, kernel, 2)
        edge = dilate - mask
        # erosion = cv2.erode(dilate, kernel, iterations=1)

    image_debug = image.copy()

    image_debug[mask>0] = image_debug[mask>0] * 0.5
    if ours_best_index != 0:
        image_debug[edge>0] = np.array([0,0,255])
    ax2.imshow(image_debug[...,::-1])
    
    auc = metrics.auc(x, insertion_ours_images_input_results)
    ax3.set_title('Insertion {:.4f}'.format(auc), fontsize=54)
    
def main(args):
    json_abs_root_file = args.json_dir
    npy_abs_root_file = args.npy_dir
    image_abs_path_root = args.image_dir
    
    makedirs(args.save_dir)
    
    class_ids = os.listdir(json_abs_root_file)
    
    for class_id in class_ids:
        json_root_file = os.path.join(json_abs_root_file, class_id)
        npy_root_file = os.path.join(npy_abs_root_file, class_id)
        image_path_root = os.path.join(image_abs_path_root, class_id)
    
        json_files = os.listdir(json_root_file)
        
        visualization_save_root_path = os.path.join(args.save_dir, class_id)
        mkdir(visualization_save_root_path)
        
        for json_file in tqdm(json_files):
            json_file_path = os.path.join(json_root_file, json_file)
            npy_file_path = os.path.join(npy_root_file, json_file.replace(".json", ".npy"))
            image_path = os.path.join(image_path_root, json_file.replace(".json", ".jpg"))
            
            visualization_save_path = os.path.join(visualization_save_root_path, json_file.replace(".json", ".png"))
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                saved_json_file = json.load(f)
                
            image = cv2.imread(image_path)
            image = cv2.resize(image, (img_size,img_size))
            attribution_map = np.load(npy_file_path)
            
            im, heatmap = gen_cam(image, norm_image(attribution_map))
            
            im = cv2.resize(im, (img_size,img_size))
            visualization(image, attribution_map, saved_json_file, im, attr_name = args.attr_name)
            plt.savefig(visualization_save_path, bbox_inches='tight',pad_inches=0.0)
            plt.clf()
            plt.close()
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
