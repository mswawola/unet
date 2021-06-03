import os
import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

CHECKPOINTS = 'checkpoints/'

def check_env():
    """
    Check PyTorch version and CUDA availability
    """
    print(f'CUDA: {torch.cuda.is_available()} - PyTorch: {torch.__version__}')

    
def save_checkpoint(state, epoch, verbose=0):
    """
    Save checkpoints
    """
    try:
        os.mkdir(CHECKPOINTS)
        print('Created checkpoint directory')
    except OSError:
        pass
    torch.save(state, CHECKPOINTS + f'BEST_CP_epoch{epoch + 1}.pth')
    if verbose == 1:
        print(f'Best checkpoint {epoch + 1} saved !')
    
    
def load_checkpoint(filename, model, verbose=0):
    """
    Load checkpoints
    """
    checkpoint = torch.load(CHECKPOINTS + filename)
    model.load_state_dict(checkpoint["state_dict"])
    if verbose == 1:
        print(f'Checkpoint {epoch + 1} loaded !')
    

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


def check_dice_score(loader, model, device="cuda"):
    """
    Compute the dice score
    """
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    dice_score = dice_score / len(loader)
    model.train()
    
    return dice_score


def evaluate(loader, model, loss_fn, device="cuda"):
    loss = 0
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.float().unsqueeze(1).to(device=device)
        with torch.no_grad():
            predictions = model(x)
            loss = loss + loss_fn(predictions, y).item()

    model.train()
    
    return loss / len(loader)

    
def save_predictions_to_tensorboard(loader, model, epoch, writer, device="cuda"):
    model.eval() #https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        writer.add_images('images', x, epoch)
        writer.add_images('masks/true', y.float().unsqueeze(1), epoch)
        writer.add_images('masks/pred', preds, epoch)

    model.train()

    
def save_results(loader, jeu, model, epoch, writer, device="cuda"):
    """
    Save results (Dice score and prdictions)
    """
    dice_score = 0
    model.eval()
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device).unsqueeze(1)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        dice_score += (2 * (preds * y).sum()) / (
            (preds + y).sum() + 1e-8)
        writer.add_images(f'images/{jeu}', x, idx)
        writer.add_images(f'masks/{jeu}/true', y.float(), idx)
        writer.add_images(f'masks/{jeu}/pred', preds, idx)
    
    dice_score = dice_score / len(loader)
    writer.add_scalar(f'dice/{jeu}', dice_score, epoch)
    
    model.train()
    
    