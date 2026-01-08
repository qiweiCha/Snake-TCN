import os  
import numpy as np  
import torch  
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.augmentations import ElasticDeform, VerticalFlip,ToTensor, RandomScale,RandomRotate,HorizontalFlip,clahe_equalized,adjust_gamma,visualize_batch
import cv2  
import torch.nn.functional as F  
from torchvision.utils import make_grid


def image_seq(images_path):
    
    files = list(sorted(os.listdir(images_path)))
 
   
    sequence_images = {}
    seq_list=[]
    id_list = [] # 序列ID列表
   
    for file in files:
       
        sequence_id = "_".join(file.split("_")[:2])
        
        image = cv2.imread(os.path.join(images_path, file), 0)
       
        image = image.astype(np.float32) / 255.0
      
        if sequence_id not in sequence_images:
            sequence_images[sequence_id] = [image]
            id_list.append(sequence_id)
        else:
            sequence_images[sequence_id].append(image)
    
    for id in id_list:
        seq = np.stack(sequence_images[id], axis=0)
        seq_list.append(seq)
            
    return seq_list, id_list


def read_image(images_path, label_path):
    
    images = []  
    gts = []  

    imagesId_list=[] 
    images,imagesId_list= image_seq(images_path)
    for i in imagesId_list:
        gt= cv2.imread(os.path.join(label_path, f"{i}.png"),cv2.IMREAD_GRAYSCALE)
        gt = np.where(gt >= 100, 1.0, 0.0)
        gt = gt[np.newaxis]   
        gts.append(gt)  
    return images, gts

def get_patch(image_list, patch_size, stride):
    patch_list = []
    sample_image = image_list[0]
    if not isinstance(sample_image, np.ndarray):
        image_list = [np.array(img) for img in image_list] 
    
    _, h, w = image_list[0].shape 
    pad_h = stride - (h - patch_size[0]) % stride
    pad_w = stride - (w - patch_size[1]) % stride
    
    for image in image_list:
            
        image_tensor = torch.from_numpy(image).float()
        image_padded = F.pad(image_tensor, (0, pad_w, 0, pad_h), "constant", 0)
        unfolded = image_padded.unfold(1, patch_size[0], stride).unfold(2, patch_size[1], stride)
        permuted = unfolded.permute(1, 2, 0, 3, 4)
        reshaped = permuted.contiguous().view(-1, permuted.shape[2], patch_size[0], patch_size[1])
        
        for sub in reshaped:
            patch_list.append(sub)
    return patch_list
class DSCA_dataset(Dataset):
    def __init__(self,  images_path, labels_path,stride,size=None,is_train=True):

        self.images_path = images_path 
        self.labels_path = labels_path 
        self.augmentations = is_train 
        self.size = size
        self.stride=stride
        # image_list[i]=[8,H,W] , gt_list[i]=[1,H,W]
        self.image_list ,self.gt_list = read_image(self.images_path, self.labels_path)    
        self.image_list=get_patch(self.image_list,self.size,self.stride)
        self.gt_list=get_patch(self.gt_list,self.size,self.stride)

        img_seed = np.random.RandomState(123)
        gt_seed = np.random.RandomState(123)


        if self.augmentations == True:
            
            self.img_transform = transforms.Compose([
                RandomRotate(img_seed, p=0.5),
                RandomScale(img_seed, p=0.5),
                # ElasticDeform(img_seed, p=0.2),
                VerticalFlip(img_seed, p=0.5),
                HorizontalFlip(img_seed, p=0.5),
                ToTensor(False)  
            ])
            self.gt_transform = transforms.Compose([
                RandomRotate(gt_seed, p=0.5),
                RandomScale(gt_seed, p=0.5),
                # ElasticDeform(gt_seed, p=0.2),
                VerticalFlip(gt_seed, p=0.5),
                HorizontalFlip(gt_seed, p=0.5),
                ToTensor(False) 
            ])
        else:

            self.img_transform = transforms.Compose([
                ToTensor(False)                      
                ])

            self.gt_transform = transforms.Compose([
                ToTensor(False)                       
                ])
        
    def __getitem__(self, idx):
        img = self.image_list[idx]  
        gt = self.gt_list[idx]  
        
        img = self.img_transform(img)
        gt = self.gt_transform(gt)
    
        return img, gt
        

    def __len__(self):
        return len(self.image_list)
   

def get_loader(images_path, labels_path, batch_size, stride,size, shuffle=True, num_workers=8, pin_memory=True, is_train=True):
    if is_train:
       
        dataset = DSCA_dataset(
            images_path=images_path, labels_path=labels_path,size=size,is_train=True,stride=stride)
    else:
         
        dataset = DSCA_dataset(
            images_path=images_path, labels_path=labels_path,size=size,is_train=False,stride=stride)
    

    loader = DataLoader(
        dataset,
        batch_size=batch_size,  
        num_workers=num_workers,  
        pin_memory=pin_memory,  
        shuffle=shuffle  
    )

    return loader
    