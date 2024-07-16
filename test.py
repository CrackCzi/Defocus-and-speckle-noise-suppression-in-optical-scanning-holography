import torch
from torchvision import transforms
import time
import torchmetrics as tm
from model import *
import numpy as np
import glob
import cv2 as cv
import matplotlib.pyplot as plt

SIZE=128
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def test(model,fn):
    """
     @brief Tests the model on an image. This is a helper function for test_fit.
     @param model The model to test.
     @param fn The filename of the image
    """
    img = cv.imread(fn)
    
    h,w,c = img.shape
    h_pad = (SIZE - h % SIZE) if h % SIZE>0 else 0
    w_pad = (SIZE - w % SIZE) if w % SIZE>0 else 0
    
    padded_img = np.empty((h+h_pad,w+w_pad,c),dtype=np.uint8)
    padded_img[:h_pad//2,:w_pad//2] = 0
    padded_img[-h_pad//2:,-w_pad//2:] = 0
    # Pads the image with the given width and height.
    if w_pad == 0 and h_pad==0:
        padded_img = img
    elif w_pad ==0:
        padded_img[h_pad//2:-h_pad//2,:,:]=img
    elif h_pad == 0:
        padded_img[:,w_pad//2:-w_pad//2,:]=img
    else:
        padded_img[h_pad//2:-h_pad//2,w_pad//2:-w_pad//2,:]=img
    tensor_img = transforms.ToTensor()(padded_img)
    blocks = []
    # Divide the image in blocks of windows size (SIZE)
    for i in range((h+h_pad)//SIZE):
        for j in range((w+w_pad)//SIZE):
            blocks.append(tensor_img[:,i*SIZE:(i+1)*SIZE,j*SIZE:(j+1)*SIZE])
    blocks = torch.stack(blocks)
    deblurred = torch.empty_like(blocks).to(device)
    with torch.no_grad():
        # This function will be called by the block manager to deblur the blocks.
        for i,block in enumerate(blocks):
            start = time.time()
            deblurred[i] = model(block.unsqueeze(0).to(device))
            print(f"INFERENCE TIME: {np.round(time.time()-start,3)}")
    for i in range((h+h_pad)//SIZE):
        for j in range((w+w_pad)//SIZE):
            k=j+(i*(w+w_pad)//SIZE)
            tensor_img[:,i*SIZE:(i+1)*SIZE,j*SIZE:(j+1)*SIZE]=deblurred[k]
    ret_img = tensor_img.permute((1,2,0)).numpy()
    ret_img = np.clip(ret_img * 255,0,255)
    out_path = "".join([*fn.split(".")[:-1],"_deblurred",".png"])
    __ = cv.imwrite(out_path,ret_img.astype(np.uint8))


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model|-m",dest="ckpt_path",default="checkpoints/best_model.pth",help="Path of the weight file")
    parser.add_argument("--input|-i",dest="dir",help="Input directory where the input images are stored")
    img_filenames = glob.glob("test/*bmp")
    ckpt_path = "checkpoints/UNet_1k1-1_mixed_lr0.0008_min1e-7_decay0.09_batch16_charbonnier.pth"
    torch.cuda.empty_cache()
    model = Unet(3,3)
    model.load_state_dict(torch.load(ckpt_path),strict=False)
    model.to(device)
    model.eval()
    # test for each file in img_filenames
    for fn in img_filenames:
        test(model,fn)