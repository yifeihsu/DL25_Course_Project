import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
import math
import random
import numpy as np

from model import ModifiedResNet

############################
# 1) MixUp / CutMix Utilities
############################

def mixup_data(x, y, alpha=1.0):
    """Compute MixUp data. Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss given the original criterion."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    """Generate a random bbox for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def cutmix_data(x, y, alpha=1.0):
    """Compute CutMix data. Returns cut-mixed inputs, pairs of targets, and lam."""
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    x1, y1 = x, y
    x2, y2 = x[index, :], y[index]

    # random bbox
    x1_coord, y1_coord, x2_coord, y2_coord = rand_bbox(x.size(), lam)

    # replace
    x1[:, :, x1_coord:x2_coord, y1_coord:y2_coord] = \
        x2[:, :, x1_coord:x2_coord, y1_coord:y2_coord]

    # adjust lambda to match the exact area of the cut region
    cut_area = (x2_coord - x1_coord) * (y2_coord - y1_coord)
    lam = 1.0 - cut_area / (x.size(-1) * x.size(-2))

    return x1, y1, y2, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """Compute CutMix loss given the original criterion."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

################################
# 2) Main Script to Resume Training
################################

if __name__ == '__main__':
    cudnn.benchmark = True

    # 1) Data Transform
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    batch_size = 256

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 2) Model
    model = ModifiedResNet(
        num_blocks=[4, 4, 3],
        base_channels=64,
        num_classes=10,
        use_se=True
    ).cuda()

    # 3) Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    # 4) Optimizer & LR scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    def lr_lambda(epoch):
        # warm-up for 10 epochs, then cosine
        warmup_epochs = 10
        total_epochs = 550  # original total
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1.0 + math.cos(
                math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            ))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(device='cuda')

    # ======================================
    #  Load from checkpoint at 550th epoch
    # ======================================
    checkpoint = torch.load('checkpoint.pth')   # Your saved checkpoint at epoch=550
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # e.g. 550 + 1 = 551
    best_acc = checkpoint['best_acc']

    print(f"Resuming training from epoch {start_epoch}, best accuracy so far: {best_acc:.2f}%")

    # 5) Hyperparams for MixUp / CutMix
    mixup_alpha = 1.0
    cutmix_alpha = 1.0
    mix_prob = 1.0

    # We want to train 200 more epochs beyond 550
    additional_epochs = 200
    end_epoch = start_epoch + additional_epochs  # e.g., 551 -> 750

    for epoch in range(start_epoch, end_epoch):
        model.train()
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()

            # Mixing or not?
            with autocast(device_type='cuda'):
                if random.random() < mix_prob:
                    # 50/50 chance of MixUp or CutMix
                    if random.random() < 0.5:
                        mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                        logits = model(mixed_images)
                        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                    else:
                        mixed_images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)
                        logits = model(mixed_images)
                        loss = cutmix_criterion(criterion, logits, targets_a, targets_b, lam)
                else:
                    # No mixing
                    logits = model(images)
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Step the LR scheduler
        scheduler.step()

        # Evaluate on test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad(), autocast(device_type='cuda'):
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                logits = model(images)
                _, preds = logits.max(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch}/{end_epoch - 1}, Test Acc: {acc:.2f}%, Best: {best_acc:.2f}%")

    print(f"Additional training done. Final epoch was {end_epoch - 1}. Best Acc = {best_acc:.2f}%")

    # Save a new checkpoint at the end of extended training
    torch.save({
        'epoch': end_epoch - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_acc': best_acc,
    }, 'checkpoint_extended.pth')
