# -- coding: utf-8 --**

"""
Created on 2024/08/22

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
                        default='submodular_results/vggsound-languagebind-efficientv2/sound-0.01-0.05-20.0-5.0-pending-samples-20-divison-10',
                        help='Save path for saliency maps generated by our methods.')
    args = parser.parse_args()
    return args

def main(args):
    print(args.explanation_dir)
    
    class_ids = os.listdir(os.path.join(args.explanation_dir, "json"))

    insertion_aucs = []
    deletion_aucs = []

    json_root_file = os.path.join(args.explanation_dir, "json")

    for class_id in tqdm(class_ids):
        json_id_files_path = os.path.join(json_root_file, class_id)

        json_file_names = os.listdir(json_id_files_path)
        for json_file_name in json_file_names:
            json_file_path = os.path.join(json_id_files_path, json_file_name)

            with open(json_file_path, 'r', encoding='utf-8') as f:
                saved_json_file = json.load(f)            

            insertion_area = [j/len(saved_json_file["consistency_score"]) for j in range(len(saved_json_file["consistency_score"])+1)]
            deletion_area = [j/len(saved_json_file["consistency_score"]) for j in range(len(saved_json_file["consistency_score"])+1)]

            insertion_score = saved_json_file["consistency_score"]
            deletion_score = saved_json_file["collaboration_score"]
            
            insertion_score = np.array([saved_json_file["baseline_score"]] + insertion_score)
            deletion_score = 1 - np.array([1-saved_json_file["org_score"]] + deletion_score)

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