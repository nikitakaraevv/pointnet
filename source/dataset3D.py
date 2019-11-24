from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
import random
import numpy as np
import utils


class Data3D(Dataset):
    def __init__(self, root_dir, classes, valid=False, folder = "train"):
        self.root_dir = root_dir
        self.categories = os.listdir(root_dir)
        self.files = []
        self.classes = classes
        self.valid=valid
        for l in self.categories:
            newdir = root_dir +'/' + l + '/' + folder + '/' ##attention
            for file in os.listdir(newdir):
                if file.endswith('.txt'):
                    o = {}
                    o['img_path'] = newdir + file
                    o['category'] = l
                    self.files.append(o)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        file.readline()
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        sampled_points = utils.sample_points(np.array(verts), faces)
        sampled_points = utils.cent_norm(sampled_points)
        if not self.valid:
            theta = random.random()*360
            sampled_points = utils.rotation_z(utils.add_noise(sampled_points), theta)
            
        return np.array(sampled_points, dtype="float32")

    def __getitem__(self, idx):
        img_path = self.files[idx]['img_path']
        category = self.files[idx]['category']
        with open(img_path, 'r') as f:
            image = self.__preproc__(f)
        return {'image': image, 'category': self.classes[category]}
