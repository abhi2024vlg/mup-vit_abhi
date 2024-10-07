### This is Flexible_ViT_attempt_patch_size_available =(8,16,32)

### Necessary Imports and dependencies
!pip install einops
import os
import shutil
import time
import math
from enum import Enum
from functools import partial
from collections import OrderedDict
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision.transforms import v2
import torchvision.transforms as transforms
from typing import Any, Dict, Union, Type, Callable, Optional, List, Tuple, Sequence
from torchvision.models.vision_transformer import MLPBlock
import wandb
import json
from PIL import Image
from torch.utils.data import Dataset
import collections
from itertools import repeat
import numpy as np

#No. of Epochs
num_epochs=30

#Warmp_Steps
warmup_try=80000

# Parameters specific to ImageNet100
batch_size = 32

# Dataset loading code
# ImageNet image size (256,256)

class ImageNet100Dataset(Dataset):
    def __init__(self, root_dirs, labels_file, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_to_idx = {}  # Map from string label to numerical index
        
        # Load labels
        with open(labels_file, 'r') as f:
            label_dict = json.load(f)
        
        # Create label to index mapping
        unique_labels = sorted(label_dict.keys())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Load image paths and labels
        for root_dir in root_dirs:
            for label in os.listdir(root_dir):
                label_path = os.path.join(root_dir, label)
                if os.path.isdir(label_path):
                    for img_name in os.listdir(label_path):
                        img_path = os.path.join(label_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(self.label_to_idx[label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor
        label = torch.tensor(label)
        
        return image, label

# Define transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.05, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the datasets
train_dirs = [
    '/kaggle/input/imagenet100/train.X1',
    '/kaggle/input/imagenet100/train.X2',
    '/kaggle/input/imagenet100/train.X3',
    '/kaggle/input/imagenet100/train.X4'
]
val_dir = ['/kaggle/input/imagenet100/val.X']
labels_file = '/kaggle/input/imagenet100/Labels.json'

train_dataset = ImageNet100Dataset(
    root_dirs=train_dirs,
    labels_file=labels_file,
    transform=train_transform
)

val_dataset = ImageNet100Dataset(
    root_dirs=val_dir,
    labels_file=labels_file,
    transform=val_transform
)

n = len(train_dataset)

total_steps = round((n * num_epochs) / batch_size)

start_step=0

mixup = v2.MixUp(alpha=0.2, num_classes=100)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: mixup(*torch.utils.data.default_collate(batch)), 
    drop_last=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
)

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, sin, cos):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def create_rope_embeddings(dim, seq_length, device, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    seq_idx = torch.arange(seq_length, device=device).float()
    sincos = torch.einsum('i,j->ij', seq_idx, theta)
    sin, cos = torch.sin(sincos), torch.cos(sincos)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    return sin, cos

class RotaryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sin, cos, attention_mask=None):
        B, N, C = x.shape
        
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_pos_emb(q, k, sin, cos)
        
        attn = (q @ k.transpose(-2, -1)) * self.scaling
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn + attention_mask

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        
        return x

class MLPBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x

class EncoderBlock(nn.Module):
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

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = RotaryAttention(hidden_dim, num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.ln_1(input)
        x = self.self_attention(x, sin, cos, attention_mask)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class Encoder(nn.Module):
    def __init__(
        self,
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
        self.layers = nn.ModuleList([
            EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            ) for _ in range(num_layers)
        ])
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.dropout(input)
        for layer in self.layers:
            x = layer(x, sin, cos, attention_mask)
        return self.ln(x)

class FlexViT(nn.Module):
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
        num_classes: int = 100,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def create_fractal_attention_mask(self, n, patch_size):
        # Calculate the number of patches in each dimension
        grid_size = int(math.sqrt(n))
        
        # Calculate the summary grid size (equivalent to 4x4 for patch_size=16)
        summary_ratio = 16 // patch_size
        summary_grid_size = grid_size // summary_ratio
        
        # Create the main grid mask
        mask_main = torch.ones(n, n)
        
        # Create the summary grid mask
        summary_size = summary_grid_size * summary_grid_size
        mask_summary = torch.ones(summary_size, summary_size)
        
        # Create the global token mask
        mask_global = torch.ones(1, 1)
        
        # Combine masks
        mask = torch.block_diag(mask_main, mask_summary, mask_global)
        
        # Connect summary tokens to their corresponding patches
        for i in range(summary_size):
            start_row = i * (summary_ratio * summary_ratio)
            end_row = (i + 1) * (summary_ratio * summary_ratio)
            mask[n + i, start_row:end_row] = 1
        
        # Connect global token to all other tokens
        mask[-1, :] = 1
        mask[:, -1] = 1
        
        # Convert to attention mask format
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        return mask

    def _process_input(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        
        n, c, h, w = x.shape
        p = patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h, n_w = h // p, w // p

        # Project patches
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h, n_w)
        x = x.permute(0, 2, 3, 1).reshape(n, n_h * n_w, self.hidden_dim)

        # Add summary tokens (scaled based on patch size)
        summary_ratio = 16 // patch_size
        summary_size = (n_h // summary_ratio) * (n_w // summary_ratio)
        summary_tokens = torch.zeros((n, summary_size, self.hidden_dim), device=x.device)
        global_token = torch.zeros((n, 1, self.hidden_dim), device=x.device)
        
        x = torch.cat([x, summary_tokens, global_token], dim=1)

        return x, n_h * n_w, summary_size

    def forward(self, x: torch.Tensor, patch_size: int):
        x, n, summary_size = self._process_input(x, patch_size)
        
        attention_mask = self.create_fractal_attention_mask(n, patch_size)
        attention_mask = attention_mask.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.device)
        
        sin, cos = create_rope_embeddings(self.hidden_dim // self.encoder.layers[0].num_heads, x.shape[1], x.device)
        
        x = self.encoder(x, sin, cos, attention_mask=attention_mask)
        
        x = x[:, -1]
        x = self.heads(x)
        
        return x

# Usage example
model = FlexViT(
    image_size=256,
    patch_size=16,
    num_layers=12,
    num_heads=6,
    hidden_dim=384,
    mlp_dim=1536,
)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model.to(device)

# Utility function for weight decay
def weight_decay_param(n, p):
    return p.ndim >= 2 and n.endswith('weight')

# Set up optimizer and learning rate scheduler
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

def save_checkpoint(state, is_best, path, filename='imagenet_baseline_patchconvcheckpoint.pth.tar'):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))

def save_checkpoint_step(step, model, best_acc1, optimizer, scheduler, checkpoint_path):
    # Define the filename with the current step
    filename = os.path.join(checkpoint_path, f'Flex_VIT.pt')
    
    # Save the checkpoint
    torch.save({
        'step': step,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, filename)

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
        
        # Convert probabilities to class indices if required (e.g., in MixUp or CutMix)
        if class_prob:
            target = target.argmax(dim=1)  # Convert from probabilities to class indices

        # Get the top-k predictions from the model output
        _, pred = output.topk(maxk, 1, True, True)  # top-k predictions for each sample
        pred = pred.t()  # Transpose pred to shape [k, batch_size]
        
        # Ensure target is reshaped to [batch_size] (if not already)
        if target.dim() > 1:
            target = target.argmax(dim=1)  # Convert target to indices if needed
            
        # Now, compare predictions with target
        correct = pred.eq(target.view(1, -1))  # Compare without unnecessary expand
        
        # Compute accuracy for top-k predictions
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Correct top-k predictions
            res.append(correct_k.mul_(1.0 / batch_size))  # Normalize by batch size
            
        return res



log_steps = 2500

wandb.login(key="cbecbe8646ebcf42a98992be9fd5b7cddae3d199")

# Initialize a new run
wandb.init(project="fractual_transformer", name="FlexViT_Modified_baseline_with_17_registers")

def validate(val_loader, model, criterion, step, patch_sizes=(8, 16, 32), use_wandb=False, print_freq=100):
    results = {}
    
    for patch_size in patch_sizes:
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':.4e', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix=f'Test (Patch Size {patch_size}): ')

        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                elif torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')

                # compute output with specific patch size
                output = model(images, patch_size=patch_size)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i)

        progress.display_summary()
        
        results[patch_size] = {
            'loss': losses.avg,
            'acc@1': top1.avg,
            'acc@5': top5.avg,
        }

    if use_wandb:
        log_data = {f'val/loss_p{p}': v['loss'] for p, v in results.items()}
        log_data.update({f'val/acc@1_p{p}': v['acc@1'] for p, v in results.items()})
        log_data.update({f'val/acc@5_p{p}': v['acc@5'] for p, v in results.items()})
        wandb.log(log_data, step=step)

    return results

def train(train_loader, val_loader, start_step, total_steps, original_model, model, criterion, optimizer, scheduler, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    print_freq = 100  # Print frequency (adjust as needed)
    log_steps = 2500  # Log steps (adjust as needed)
    
    progress = ProgressMeter(
        total_steps,
        [batch_time, data_time, losses, top1, top5]
    )

    model.train()
    end = time.time()
    best_acc1 = 0
    
    
    def infinite_loader():
        while True:
            yield from train_loader

    patch_sizes = [8, 16, 32]  # Available patch sizes

    for step, (images, target) in zip(range(start_step + 1, total_steps + 1), infinite_loader()):
        
        
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Randomly choose patch size for this step
        patch_size = np.random.choice(patch_sizes)

        output = model(images, patch_size=patch_size)
        loss = criterion(output, target)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))
        
        loss.backward()
        l2_grads = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if step % print_freq == 0:
            progress.display(step)
            if wandb:
                with torch.no_grad():
                    l2_params = sum(p.square().sum().item() for _, p in model.named_parameters())
                
                samples_per_second_per_gpu = batch_size / batch_time.val
                samples_per_second = samples_per_second_per_gpu 
                log_data = {
                    "train/loss": loss.item(),
                    'train/acc@1': acc1[0].item(),
                    'train/acc@5': acc5[0].item(),
                    "data_time": data_time.val,
                    "batch_time": batch_time.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr": scheduler.get_last_lr()[0],
                    "l2_grads": l2_grads.item(),
                    "l2_params": math.sqrt(l2_params),
                    "patch_size": patch_size
                }
                wandb.log(log_data, step=step)
        
        if step % print_freq == 0 and step % log_steps != 0 and step != total_steps:
            save_checkpoint_step(step, model, best_acc1, optimizer, scheduler, checkpoint_path)

                
        if step % log_steps == 0 or step == total_steps:
            val_results = validate(val_loader, original_model, criterion, step, patch_sizes=patch_sizes, use_wandb=True)
            
            # Use the average acc@1 across all patch sizes for determining the best model
            avg_acc1 = sum(result['acc@1'] for result in val_results.values()) / len(val_results)
            is_best = avg_acc1 > best_acc1
            best_acc1 = max(avg_acc1, best_acc1)
            
            save_checkpoint({
                'step': step,
                'state_dict': original_model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, checkpoint_path)

        scheduler.step()
        
        if step % 40000 == 0 and step > 0:
            break

train(train_loader, val_loader, start_step, total_steps, original_model, model, criterion, optimizer, scheduler, device)

wandb.finish()
