import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def patches_to_tensors(from_patch, to_patch):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (H x W)
    to a torch.FloatTensor of shape (H x W)
    
    Args:
         from_patch (numpy.ndarray): Patch from the first image.
         to_patch (numpy.ndarray): Patch from the second image.
    Returns:
         Tensor1: Converted first patch.
         Tensor2: Converted second patch.
    """
    return torch.from_numpy(from_patch), torch.from_numpy(to_patch)

def load_patches_image(from_cam, to_cam, root_dir, frame_name, from_top_LeftX, from_top_LeftY, to_top_LeftX, to_top_LeftY):
    from_img = cv2.imread(root_dir + 'cam' + str(from_cam) + '/' + frame_name)
    to_img = cv2.imread(root_dir + 'cam' + str(to_cam) + '/' + frame_name)
    from_patch = from_img[from_top_LeftX:from_top_LeftX+96, from_top_LeftY:from_top_LeftY+96]
    to_patch = to_img[to_top_LeftX:to_top_LeftX+96, to_top_LeftY:to_top_LeftY+96]   
    return (from_patch, to_patch)

def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start+num):
        img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        img = (img/255.)*2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start+num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w,h = imgx.shape
        if w < 224 or h < 224:
            d = 224.-min(w,h)
            sc = 1+d/min(w,h)
            imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
            imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)

        imgx = (imgx/255.)*2 - 1
        imgy = (imgy/255.)*2 - 1
        img = np.asarray([imgx, imgy]).transpose([1,2,0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, mode, split = 'train'):
    count_items = 0
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    for frameName in data.keys():
        #if data[frame]['subset'] != split:
            #continue

        #if not os.path.exists(os.path.join(root, frameName)):
            #continue

        num_pairs = len(data[frameName])
        #if mode == "flow":
            #num_frames = num_frames//2
        
        #fps = num_frames/data[vid]['duration']
        
        for j in range(num_pairs):
            from_cam, to_cam = data[frameName][j]['fromCam'], data[frameName][j]['toCam']
            from_top_LeftX, from_top_LeftY = data[frameName][j]['fromTopLeftX'], data[frameName][j]['fromTopLeftY']
            to_top_LeftX, to_top_LeftY = data[frameName][j]['toTopLeftX'], data[frameName][j]['toTopLeftY']

            #if j+snippets>num_frames:
                #continue
            #label = np.zeros((num_classes, snippets), np.float32)
            #for ann in data[vid]['actions']:
                #for fr in range(j+1,j+snippets+1,1):
                    #if fr/fps >= ann[1] and fr/fps <= ann[2]:
                        #label[ann[0], (fr-1)%snippets] = 1
            label = np.array([data[frameName][j]['rDist'], data[frameName][j]['phi'], data[frameName][j]['theta']])
            dataset.append((frameName, from_cam, to_cam, from_top_LeftX, from_top_LeftY, to_top_LeftX, to_top_LeftY, j+1, label))
            count_items += 1

    print("Make dataset {}: {} examples".format(split, count_items))
    
    return dataset


class MVOR(data_utl.Dataset):

    def __init__(self, split_file, split, mode='spheric', transforms=None):
        
        self.data = make_dataset(split_file, mode, split = 'train')
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root_dir = '/Users/hamoudmohamed/Desktop/MVORPartial/color/'
        self.N = 18119
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        frame_name, from_cam, to_cam, from_top_LeftX, from_top_LeftY, to_top_LeftX, to_top_LeftY, start, label = self.data[index]
        #start_f = random.randint(1,nf-65)

        #if self.mode == 'rgb':
            #imgs = load_rgb_frames(self.root, vid, start, self.snippets)
        #else:
            #imgs = load_flow_frames(self.root, vid, start, self.snippets)
        #label = label[:, :] #start_f:start_f+64]
        from_patch, to_patch = load_patches_image(from_cam, to_cam, self.root_dir, frame_name, from_top_LeftX, from_top_LeftY, to_top_LeftX, to_top_LeftY)
        #imgs = self.transforms(imgs)

        return patches_to_tensors(from_patch, to_patch)[0], patches_to_tensors(from_patch, to_patch)[1], torch.from_numpy(label)

    def __len__(self):
        return len(self.data)




class MVOR_eval(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, snippets, transforms=None):
        
        self.data = make_dataset(split_file, split, root, mode, snippets)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        frameName, from_cam, to_cam, from_top_LeftX, from_top_LeftY, to_top_LeftX, to_top_LeftY, start, label = self.data[index]
        #start_f = random.randint(1,nf-65)

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start, self.snippets)
        else:
            imgs = load_flow_frames(self.root, vid, start, self.snippets)
        #label = label[:, :] #start_f:start_f+64]
        
        imgs = self.transforms(imgs)

        return vid, start, video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)