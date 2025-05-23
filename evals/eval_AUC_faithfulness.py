# -- coding: utf-8 --**

"""
Created on 2023/12/1

@author: Ruoyu Chen
"""

import argparse

import os
import os
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from tqdm import tqdm

from sklearn import metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Faithfulness Metric')
    parser.add_argument('--explanation-dir', 
                        type=str, 
                        default='submodular_results_iclr_baseline/imagenet-languagebind-false/grad-10x10-4',
                        help='Save path for saliency maps generated by our methods.')
    args = parser.parse_args()
    return args

def main(args):
    print(args.explanation_dir)
    
    class_ids = os.listdir(os.path.join(args.explanation_dir, "npy"))

    insertion_aucs = []
    deletion_aucs = []

    json_root_file = os.path.join(args.explanation_dir, "json")
    npy_root_file = os.path.join(args.explanation_dir, "npy")

    for class_id in tqdm(class_ids):
        json_id_files_path = os.path.join(json_root_file, class_id)
        npy_id_files_path = os.path.join(npy_root_file, class_id)

        json_file_names = os.listdir(json_id_files_path)
        for json_file_name in json_file_names:
            json_file_path = os.path.join(json_id_files_path, json_file_name)
            npy_file_path = os.path.join(npy_id_files_path, json_file_name.replace(".json", ".npy"))

            with open(json_file_path, 'r', encoding='utf-8') as f:
                saved_json_file = json.load(f)
            submodular_image_set = np.load(npy_file_path)

            insertion_area = []
            deletion_area = []
            image = submodular_image_set.sum(0)

            insertion_ours_image = image.copy() - image.copy() # baseline
            deletion_ours_image = image.copy()                  # full image

            insertion_area.append(
                (insertion_ours_image.sum(-1)!=0).sum() / (image.shape[0] * image.shape[1]))
            deletion_area.append(
                (deletion_ours_image.sum(-1)!=0).sum() / (image.shape[0] * image.shape[1]))


            for smdl_sub_mask in submodular_image_set:
                insertion_ours_image = insertion_ours_image + smdl_sub_mask
                deletion_ours_image = image - insertion_ours_image
                
                insertion_area.append(
                    (insertion_ours_image.sum(-1)!=0).sum() / (image.shape[0] * image.shape[1]))
                deletion_area.append(
                    (deletion_ours_image.sum(-1)!=0).sum() / (image.shape[0] * image.shape[1]))

            insertion_score = saved_json_file["consistency_score"]
            deletion_score = saved_json_file["collaboration_score"]
            
            insertion_score = np.array([1 - deletion_score[-1]] + insertion_score)
            deletion_score = 1 - np.array([1 - insertion_score[-1]] + deletion_score)

            insertion_auc = metrics.auc(np.array(insertion_area), insertion_score)
            deletion_auc = metrics.auc(1-np.array(deletion_area), deletion_score)
            insertion_aucs.append(insertion_auc)
            deletion_aucs.append(deletion_auc)
    insertion_auc_score = np.array(insertion_aucs).mean()
    deletion_auc_score = np.array(deletion_aucs).mean()
    print("Insertion AUC Score: {:.4f}\nDeletion AUC Score: {:.4f}".format(insertion_auc_score, deletion_auc_score))
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)