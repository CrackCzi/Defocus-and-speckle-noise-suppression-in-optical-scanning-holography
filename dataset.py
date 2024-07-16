
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
import cv2


def is_image_file(filename):
    """
     @brief Checks if filename is an image file. This is a helper function for get_image_file ()
     @param filename The filename to check.
     @return True if the filename ends with one of the extensions False otherwise. For example if you want to check for image files
    """
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'bmp', 'BMP'])

def load_img(filepath):
    """
     @brief Loads an image from a file and converts it to floating point. The image is returned as a Numpy array of float32 values in [ 0 1 ]
     @param filepath Path to the image file
     @return Image as a Numpy array in [ 0 1 ] with values in [ 0 1 ]. If the image cannot be loaded an exception is
    """
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        """
         @brief Initialize class. The purpose of this method is to set up the data loader for use in training.
         @param rgb_dir Directory containing RGB images.
         @param img_options Options for image creation. 
         @param target_transform
        """
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_image_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)for x in noisy_files if is_image_file(x)]

        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        """
         @brief Return the length of the tar. This is used to determine how many files are in the archive.
         @return The length of the tar in bytes or None if there is no data in the archive
        """
        return self.tar_size

    def __getitem__(self, index):
        """
         @brief Return image at index. This is a function to be used by __getitem__. In this case index is 0 - based
         @param index Index of the image to return. Must be in range 0 to self. tar_size - 1
         @return A tuple of 2 images
        """
        tar_index   = index % self.tar_size
        clean = load_img(self.clean_filenames[tar_index])
        noisy = load_img(self.noisy_filenames[tar_index])
        

        #Crop Input and Target
        ps = 128
        H = clean.shape[0]
        W = clean.shape[1]
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)

        transformed = self.target_transform(image=clean,mask=noisy)
        clean = transformed["image"]
        noisy = transformed['mask']
        # torch.unsqueeze(noisy,1)
        # print(clean.shape)
        # print(noisy.shape)
        noisy = noisy.permute(2,0,1)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        return clean, noisy


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        """
         @brief Initialize class. The purpose of this method is to set up the data loader for use in validation.
         @param rgb_dir Directory containing RGB images. Must be a directory named groundtruth and input
         @param target_transform Transform to use for
        """
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_image_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_image_file(x)]


        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        """
         @brief Return the length of the tar. This is used to determine how many files are in the archive.
         @return The length of the tar in bytes or None if there is no data in the archive.
        """
        return self.tar_size

    def __getitem__(self, index):
        """
         @brief Return the image at the given index.
         @param index Index of the image to return. Must be in range 0 to self. tar_size - 1
         @return A tuple of 2 images
        """
        tar_index   = index % self.tar_size


        clean = load_img(self.clean_filenames[tar_index])
        noisy = load_img(self.noisy_filenames[tar_index])

        transformed = self.target_transform(image=clean,mask=noisy)
        clean = transformed["image"]
        noisy = transformed['mask']
        noisy = noisy.permute(2,0,1)
        return clean, noisy




def get_training_data(rgb_dir):
    """
     @brief Creates training data for RGB images. This is a helper function to create a data loader that can be used to train the model.
     @param rgb_dir Path to the directory containing RGB images.
     @return A DataLoaderTrain object that can be used to train the model and get the images from
    """
    # assert os.path.exists(rgb_dir)
    transforms = A.Compose([A.Rotate(),A.Flip(),ToTensorV2()])

    return DataLoaderTrain(rgb_dir, target_transform=transforms)


def get_validation_data(rgb_dir):
    """
     @brief Creates training data for RGB images. This is a helper function to create a data loader that can be used to validate the model.
     @param rgb_dir Path to the directory containing RGB images.
     @return A DataLoaderVal object that can be used to validate the model and get the images from
    """
    transforms = A.Compose([A.CenterCrop(128,128),ToTensorV2()])
    # assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, target_transform=transforms)