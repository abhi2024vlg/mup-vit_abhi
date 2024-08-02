### Necessary Imports and dependencies
### Wandb project_name is ImageNet_Avg_Conv2D
import os
import shutil
import time
import math
from enum import Enum
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
from torchvision.transforms import v2
import torchvision.transforms as transforms
from typing import Any, Dict, Union, Type, Callable, Optional, List
from torchvision.models.vision_transformer import MLPBlock
import wandb

num_epochs=90

# Parameters specific to CIFAR-10
batch_size = 128
num_workers = 4 

# Dataset loading code
# Define CIFAR-10 datasets with image size 256x256
# Needs to be replaced by ImagNet image size256,256

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.05, 1.0)),  # Change to 256x256
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
)

val_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.Resize(256),  # Resize to 256
        transforms.CenterCrop(256),  # Center crop to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
)

n = len(train_dataset)

total_steps = round((n * num_epochs) / batch_size)

start_step=0

mixup = v2.MixUp(alpha=0.2, num_classes=10)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=lambda batch: mixup(*torch.utils.data.default_collate(batch)), 
    drop_last=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

warmup_try=10000

class PatchExtractor(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_channels=3, embed_dim=384):
        super(PatchExtractor, self).__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_original = (image_size // patch_size) ** 2
        self.num_patches_downsampled_64 = ((image_size // 4) // patch_size) ** 2
        self.num_patches_downsampled_16 = 1  # 16x16 image gives 1 patch of 16x16
        
        self.projection= nn.Conv2d(in_channels=in_channels, out_channels=embed_dim,kernel_size=patch_size, stride=patch_size, padding=0, bias=False)
        
        self.downsample = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        
        self.embed_dim = embed_dim
        
    def forward(self, x):
        # x shape: (batch_size, in_channels, 256, 256)
        
        # Process original image
        x_original = self.projection(x)
        x_original = x_original.flatten(2).transpose(1, 2)
        # x_original shape: (batch_size, 256, embed_dim)
        
        # Downsample to 64x64
        x_64 = self.downsample(x)
        # x_64 shape: (batch_size, in_channels, 64, 64)
        
        # Process 64x64 image
        x_64_patches = self.projection(x_64)
        x_64_patches = x_64_patches.flatten(2).transpose(1, 2)
        # x_64_patches shape: (batch_size, 16, embed_dim)
        
        # Downsample to 16x16
        x_16 = self.downsample(x_64)
        # x_16 shape: (batch_size, in_channels, 16, 16)
        
        # Process 16x16 image
        x_16_patch = self.projection(x_16)
        x_16_patch = x_16_patch.flatten(2).transpose(1, 2)
        # x_16_patches shape: (batch_size, 1, embed_dim)
        
        # Concatenate all
        x_combined = torch.cat([x_original, x_64_patches, x_16_patch], dim=1)
        # x_combined shape: (batch_size, 273, embed_dim)
        
        return x_combined

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

        # Fix init discrepancy between nn.MultiheadAttention and that of big_vision
        bound = math.sqrt(3 / hidden_dim)
        nn.init.uniform_(self.self_attention.in_proj_weight, -bound, bound)
        nn.init.uniform_(self.self_attention.out_proj.weight, -bound, bound)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        return self.ln(self.layers(self.dropout(input)))


class SimpleVisionTransformer(nn.Module):
    """Vision Transformer modified per https://arxiv.org/abs/2205.01580."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 10, # No. of classes have to be changed to 1000 for imagenet
        seq_length: int = 273,  # 256 + 16 + 1
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        # Add PatchExtractor
        self.patch_extractor = PatchExtractor(image_size=image_size, patch_size=patch_size, in_channels=3, embed_dim=hidden_dim)
        
        # Update seq_length to match the number of tokens from TokenGenerator
        self.seq_length = seq_length
        
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)
        
        # Initialize weights for the heads
        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)


    def forward(self, x: torch.Tensor):
        # Use TokenGenerator to get tokens
        #print(x.shape)
        x = self.patch_extractor(x)  # Shape: [B,5, hidden_dim]
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)
        x = x.mean(dim = 1)
        #print(x.shape)
        x = self.heads(x)
        #print(x.shape)
        return x
    
def weight_decay_param(n, p):
    return p.ndim >= 2 and n.endswith('weight')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create model
model = SimpleVisionTransformer(
    image_size=256,
    patch_size=16,
    num_layers=12,
    num_heads=6,
    hidden_dim=384,
    mlp_dim=1536,
).to(device)
wd_params = [p for n, p in model.named_parameters() if weight_decay_param(n, p) and p.requires_grad]
non_wd_params = [p for n, p in model.named_parameters() if not weight_decay_param(n, p) and p.requires_grad]

original_model = model

weight_decay = 0.1
learning_rate = 1e-3

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(
    [
        {"params": wd_params, "weight_decay": 0.1},
        {"params": non_wd_params, "weight_decay": 0.},
    ],
    lr=learning_rate,
)

warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: step / warmup_try)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_try)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [warmup_try])

#Change_path_for_the_directory;This is the directory where model weights are to be saved
checkpoint_path = "/kaggle/working/"

def save_checkpoint(state, is_best, path, filename='imagenet_average_patchconvcheckpoint.pth.tar'):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,), class_prob=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # with e.g. MixUp target is now given by probabilities for each class so we need to convert to class indices
        if class_prob:
            _, target = target.topk(1, 1, True, True)
            target = target.squeeze(dim=1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res
    
log_steps = 2500

# Initialize a new run
wandb.init(project="fractual_transformer", name="ImageNet_AvgConv_run")

def validate(val_loader, model, criterion, step, use_wandb=False, accum_freq=1, print_freq=100):
    
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            torch.cuda.empty_cache()
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i

                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                elif torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')

                for img, trt in zip(images.chunk(accum_freq), target.chunk(accum_freq)):
                    # compute output
                    output = model(img)
                    loss = criterion(output, trt)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, trt, topk=(1, 5))
                    losses.update(loss.item(), img.size(0))
                    top1.update(acc1[0].item(), img.size(0))
                    top5.update(acc5[0].item(), img.size(0))
                    
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    if use_wandb:        
        log_data = {
            'val/loss': losses.avg,
            'val/acc@1': top1.avg,
            'val/acc@5': top5.avg,
        }
        wandb.log(log_data, step=step)

    return top1.avg

def train(train_loader, val_loader, start_step, total_steps, original_model, model, criterion, optimizer, scheduler, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    print_freq = 100  # Print frequency (adjust as needed)
    log_steps = 2500  # Log steps (adjust as needed)
    accum_freq = 1  # Gradient accumulation frequency (adjust as needed)
    
    progress = ProgressMeter(
        total_steps,
        [batch_time, data_time, losses, top1, top5]
    )

    # switch to train mode
    model.train()
    end = time.time()
    best_acc1 = 0
    
    def infinite_loader():
        while True:
            yield from train_loader

    for step, (images, target) in zip(range(start_step + 1, total_steps + 1), infinite_loader()):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        step_loss = step_acc1 = step_acc5 = 0.0

        for img, trt in zip(images.chunk(accum_freq), target.chunk(accum_freq)):
            # compute output
            output = model(img)
            loss = criterion(output, trt)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, trt, topk=(1, 5), class_prob=True)
            step_loss += loss.item()
            step_acc1 += acc1[0].item()
            step_acc5 += acc5[0].item()
            
            # compute gradient
            (loss / accum_freq).backward()

        step_loss /= accum_freq
        step_acc1 /= accum_freq
        step_acc5 /= accum_freq

        losses.update(step_loss, images.size(0))
        top1.update(step_acc1, images.size(0))
        top5.update(step_acc5, images.size(0))
        
         # do SGD step
        l2_grads = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0:
            print(step)
            progress.display(step)
            if wandb:
                
                with torch.no_grad():
                    l2_params = sum(p.square().sum().item() for _, p in model.named_parameters())

                samples_per_second_per_gpu = batch_size / batch_time.val
                samples_per_second = samples_per_second_per_gpu 
                log_data = {
                    "train/loss": step_loss,
                    'train/acc@1': step_acc1,
                    'train/acc@5': step_acc5,
                    "data_time": data_time.val,
                    "batch_time": batch_time.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr": scheduler.get_last_lr()[0],
                    "l2_grads": l2_grads.item(),
                    "l2_params": math.sqrt(l2_params)
                }
                wandb.log(log_data, step=step)

        if step % log_steps == 0 or step == total_steps:
            
            acc1 = validate(val_loader, original_model, criterion, step)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            
            save_checkpoint({
                'step': step,
                'state_dict': original_model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best,checkpoint_path)

        scheduler.step()
        

train(train_loader, val_loader, start_step, total_steps, original_model, model, criterion, optimizer, scheduler, device)
