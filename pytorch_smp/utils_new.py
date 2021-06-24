import torch
import os
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)



def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    precision_score = 0
    recall_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            precision_score += smp.utils.functional.precision(preds, y)
            recall_score += smp.utils.functional.recall(preds, y)
            

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

    acc=num_correct/num_pixels*100
    dice=dice_score/len(loader)
    recall=recall_score/len(loader)
    precision=precision_score/len(loader)
    return  acc, dice, precision, recall

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

#Jonas added this
def create_logdir(root, lr, loss_fn, network):
    logdir=root+network+"_"+loss_fn+"_"+str(lr)+"/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(logdir+"saved_images"):
        os.mkdir(logdir+"saved_images")
    
    return logdir

#Jonas added this
def write_log_file(acc, dice, epoch, logdir, precision, recall):
    f= open((logdir+"logfile.txt"), "a")
    f.write("\n")
    content="Epoch nr : "+ str(epoch)+ "    accuracy: "+ str(acc) +"   dice: " +str(dice) + "   precision: " +str(precision) + "   recall: " + str(recall)
    f.write(content)
    f.close()
