import torch
import time
import torchmetrics.image as tm
from torch.utils.data import DataLoader
from model import *
from dataset import *

# valid_dir = "dataset/eval/Speckle_eval"
valid_dir = "dataset/eval/Defocus_eval"

def eval(ckpt_path):
    """
     @brief Evaluate Unet model.
     @param ckpt_path path to the model
    """
    val_dataset = get_validation_data(valid_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True, drop_last=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    psnr = tm.PeakSignalNoiseRatio().to(device)
    ssim = tm.StructuralSimilarityIndexMeasure().to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.cuda.empty_cache()

    # initialize Unet with default (paper) parameters
    model = Unet(3, 3)
    model.load_state_dict(torch.load(ckpt_path),strict=False)
    model.to(device)

    model.eval()
    with torch.no_grad():
        psnr_scalar = 0
        ssim_scalar = 0
        j = 0
        # This function is used to generate the PSNR SSIM and PSNR values.
        for i, data in enumerate(val_loader):
            target = data[0].to(device)
            input = data[1].to(device)
            with torch.cuda.amp.autocast():
                pred = model(input)
            psnr_scalar+=psnr(pred,target).item()
            ssim_scalar+=ssim(pred,target).item()
            j += 1
        print("PSNR {:.3f}; SSIM {:.4f};".format(psnr_scalar/(j+1), ssim_scalar/(j+1)))


# checkpoint
if __name__=="__main__":
    weights = "checkpoints/UNet_1k1-1_mixed_lr0.0008_min1e-7_decay0.09_batch16_charbonnier.pth"
    eval(weights)