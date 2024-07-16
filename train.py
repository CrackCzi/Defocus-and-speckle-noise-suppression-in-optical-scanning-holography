import torch
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter
import time
import torchmetrics.image as tm
from torch.utils.data import DataLoader
from model import *
from dataset import *

EPOCHS = 60
BATCH_SIZE = 16

train_dir = "dataset/1k_mixed/train"
valid_dir = "dataset/1k_mixed/valid"


def train():
    """
     @brief Train Unet on data stored in train_dir and validate it on valid_dir.
    """
    train_dataset = get_training_data(train_dir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=False)
    val_dataset = get_validation_data(valid_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    psnr = tm.PeakSignalNoiseRatio().to(device)
    ssim = tm.StructuralSimilarityIndexMeasure().to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    loss_scaler = NativeScaler()
    torch.cuda.empty_cache()

    # initialize Unet with default (paper) parameters
    model = Unet(3, 3)
    model.to(device)
    criterion = CharbonnierLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 80e-5, weight_decay=0.09)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=1E-7)
    logger = SummaryWriter()
    best_model_psnr=0
    
    # This is the loop used to train and eval the models
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        train_loss = 0
        valid_loss = 0
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            target = data[0].to(device)
            input = data[1].to(device)
            with torch.cuda.amp.autocast():
                pred = model(input)
                loss = criterion(pred,target)
            loss_scaler(loss,optimizer,parameters=model.parameters())
            train_loss+=loss.item()
        logger.add_scalar("Train Loss",train_loss,epoch)

        model.eval()
        with torch.no_grad():
            psnr_scalar = 0
            ssim_scalar = 0
            j = 0
            for i, data in enumerate(val_loader):
                target = data[0].to(device)
                input = data[1].to(device)
                with torch.cuda.amp.autocast():
                    pred = model(input)
                loss = criterion(pred,target)
                valid_loss+=loss.item()
                psnr_scalar+=psnr(pred,target).item()
                ssim_scalar+=ssim(pred,target).item()
                j += 1
        stop = time.time() - epoch_start_time
        print("@EPOCH {}/{}: Train Loss {:.3f}; Valid Loss {:.3f}; PSNR {:.3f}; SSIM {:.3f}; Time {:.3f}"
              .format((epoch + 1), EPOCHS, train_loss, valid_loss, psnr_scalar / (j + 1), ssim_scalar / (j + 1), stop))

        # Save the model per 20 epoch and the best model of all time.
        if psnr_scalar>best_model_psnr:
            best_model_psnr=psnr_scalar
            model.cpu()
            torch.save(model.state_dict(),"model/UNet_1k1-1_mixed_lr0.0008_min1e-7_decay0.09_batch16_charbonnier.pth")
            model.to(device)
        if (epoch > 0) & (epoch % 20 == 19):
            model.cpu()
            torch.save(model.state_dict(), f"model/ckpt_{epoch+1}UNet_1k1-1_mixed_lr0.0008_min1e-7_decay0.09_batch16_charbonnier.pth")
            model.to(device)
        logger.add_scalar("PSNR",psnr_scalar/(j+1),epoch)
        logger.add_scalar("SSIM",ssim_scalar/(j+1),epoch)
        # stop = time.time() - epoch_start_time
        scheduler.step()

        logger.add_scalar("Valid Loss", valid_loss, epoch)


if __name__=="__main__":
    train()