import importlib
import random
import cv2
import numpy as np
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter, convolve,zoom
from skimage import measure
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from torchvision.utils import make_grid
import os  

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
GLOBAL_RANDOM_STATE = np.random.RandomState(47)

def clahe_equalized(img):
    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    
    if len(img.shape) == 2:  
        return clahe.apply(img)
    else:  
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)



def adjust_gamma(img, gamma=1.0):
  
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in np.arange(0, 256)]).astype("uint8")
    
    if len(img.shape) == 3:  
        return cv2.LUT(img, table)
    else: 
        return cv2.LUT(img, table)



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, m):
        for t in self.transforms:
            m = t(m)
        return m

class HorizontalFlip:
   
    def __init__(self, random_state, p=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.p = p  

    def __call__(self, m):
        assert m.ndim == 3, f'Supports only 3D (DxHxW) images, but got shape {m.shape}'

        
        if self.random_state.uniform() < self.p:
            is_tensor = isinstance(m, torch.Tensor)
            
           
            if is_tensor:
                
                return torch.flip(m, dims=[2])
            else:
               
                return np.flip(m, axis=2).copy()
        
        return m

class VerticalFlip:
   
    def __init__(self, random_state, p=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.p = p

    def __call__(self, m):
        assert m.ndim == 3, f'Supports only 3D (DxHxW) images, but got shape {m.shape}'

        if self.random_state.uniform() < self.p:
            is_tensor = isinstance(m, torch.Tensor)
            
            
            if is_tensor:
              
                return torch.flip(m, dims=[1])
            else:
               
                return np.flip(m, axis=1).copy()

        return m


class RandomRotate:
    
    def __init__(self, random_state, rotation_range=(-15, 15), p=0.5, **kwargs):
        
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.rotation_range = rotation_range
        self.p = p

    def __call__(self, m):
        assert m.ndim == 3, f'Supports only 3D (DxHxW) images, but got shape {m.shape}'

        if self.random_state.uniform() < self.p:
            is_tensor = isinstance(m, torch.Tensor)
            if is_tensor:
                m = m.numpy()

           
            angle = self.random_state.uniform(self.rotation_range[0], self.rotation_range[1])
            
           
            h, w = m.shape[1:]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
          
            rotated_m = np.zeros_like(m)
            for i in range(m.shape[0]):
               
                rotated_m[i] = cv2.warpAffine(m[i], rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            if is_tensor:
                return torch.from_numpy(rotated_m)
            return rotated_m
        
        return m
class RandomScale:
   
    def __init__(self, random_state, scale_range=(0.9, 1.1), p=0.5, **kwargs):
       
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.scale_range = scale_range
        self.p = p

    def __call__(self, m):
        assert m.ndim == 3, f'Supports only 3D (DxHxW) images, but got shape {m.shape}'

        if self.random_state.uniform() < self.p:
            is_tensor = isinstance(m, torch.Tensor)
            if is_tensor:
                m = m.numpy()

           
            scale = self.random_state.uniform(self.scale_range[0], self.scale_range[1])
            
           
            h, w = m.shape[1:]
            center = (w // 2, h // 2)
           
            scale_matrix = cv2.getRotationMatrix2D(center, 0, scale)
            
           
            scaled_m = np.zeros_like(m)
            for i in range(m.shape[0]):
                scaled_m[i] = cv2.warpAffine(m[i], scale_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            if is_tensor:
                return torch.from_numpy(scaled_m)
            return scaled_m
        
        return m


class ElasticDeform:
  
    def __init__(self, random_state, alpha=35, sigma=5, p=0.5, **kwargs):
       
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, m):
        assert m.ndim == 3, f'Supports only 3D (DxHxW) images, but got shape {m.shape}'

        if self.random_state.uniform() < self.p:
            is_tensor = isinstance(m, torch.Tensor)
            if is_tensor:
                m = m.numpy()

            h, w = m.shape[1:]

           
            dx = self.random_state.rand(h, w) * 2 - 1
            dy = self.random_state.rand(h, w) * 2 - 1

          
            ksize = int(2 * round(2.5 * self.sigma) + 1)
            if ksize % 2 == 0:
                ksize += 1
            
            dx = cv2.GaussianBlur(dx, (ksize, ksize), self.sigma) * self.alpha
            dy = cv2.GaussianBlur(dy, (ksize, ksize), self.sigma) * self.alpha
            
            
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            
          
            deformed_m = np.zeros_like(m)
            for i in range(m.shape[0]):
                deformed_m[i] = cv2.remap(m[i], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            if is_tensor:
                return torch.from_numpy(deformed_m)
            return deformed_m
        
        return m

class RandomGamma:
   
    def __init__(self, random_state, gamma_range=(0.7, 1.5), p=0.5, **kwargs):
      
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, m):
        assert m.ndim == 3, f'Supports only 3D (DxHxW) images, but got shape {m.shape}'

        if self.random_state.uniform() < self.p:
            is_tensor = isinstance(m, torch.Tensor)
            if is_tensor:
                m_np = m.numpy()
            else:
                m_np = m
            
            gamma = self.random_state.uniform(self.gamma_range[0], self.gamma_range[1])
            
            m_corrected = np.power(m_np, 1.0 / gamma)

            if is_tensor:
                
                return torch.from_numpy(m_corrected).to(m.dtype)
            return m_corrected
        
        return m


class CropToFixed:
    def __init__(self, random_state, size=(256, 256), centered=False, **kwargs):
        self.random_state = random_state
        self.crop_y, self.crop_x = size
        self.centered = centered

    def __call__(self, m):
        def _padding(pad_total):
            half_total = pad_total // 2
            return (half_total, pad_total - half_total)

        def _rand_range_and_pad(crop_size, max_size):
            """
            Returns a tuple:
                max_value (int) for the corner dimension. The corner dimension is chosen as `self.random_state(max_value)`
                pad (int): padding in both directions; if crop_size is lt max_size the pad is 0
            """
            if crop_size < max_size:
                return max_size - crop_size, (0, 0)
            else:
                return 1, _padding(crop_size - max_size)

        def _start_and_pad(crop_size, max_size):
            if crop_size < max_size:
                return (max_size - crop_size) // 2, (0, 0)
            else:
                return 0, _padding(crop_size - max_size)

        assert m.ndim in (2, 3)
        if m.ndim == 3:
            _, y, x = m.shape
        else:
            y, x = m.shape

        if not self.centered:
            y_range, y_pad = _rand_range_and_pad(self.crop_y, y)
            x_range, x_pad = _rand_range_and_pad(self.crop_x, x)

            y_start = self.random_state.randint(y_range)
            x_start = self.random_state.randint(x_range)

        else:
            y_start, y_pad = _start_and_pad(self.crop_y, y)
            x_start, x_pad = _start_and_pad(self.crop_x, x)

        # if m.ndim == 3:
        result = m[:, y_start:y_start + self.crop_y,
                   x_start:x_start + self.crop_x]
        return np.pad(result, pad_width=((0, 0), y_pad, x_pad), mode='reflect')

class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim == 3, 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        # 修改后（正确代码）
        if isinstance(m, torch.Tensor):
            return m.to(dtype=torch.float32)  # PyTorch方式转换类型
        else:
            return torch.from_numpy(np.array(m).astype(self.dtype))  # 兼容NumPy输入








