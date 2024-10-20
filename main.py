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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def to_2tuple(x: Any) -> Tuple:
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))

def resize_abs_pos_embed(
    pos_embed: torch.Tensor,
    new_size: Tuple[int, int],
    old_size: Optional[Union[int, Tuple[int, int]]] = None,
    num_prefix_tokens: int = 1,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> torch.Tensor:
    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]
    if not old_size:
        old_size = int(math.sqrt(pos_embed.shape[1] - num_prefix_tokens))
    old_size = to_2tuple(old_size)
    if new_size == old_size:
        return pos_embed
    if num_prefix_tokens:
        posemb_prefix, pos_embed = pos_embed[:, :num_prefix_tokens], pos_embed[:, num_prefix_tokens:]
    else:
        posemb_prefix, pos_embed = None, pos_embed
    pos_embed = pos_embed.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(pos_embed, size=new_size, mode=interpolation, antialias=antialias)
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)
    if posemb_prefix is not None:
        pos_embed = torch.cat([posemb_prefix, pos_embed], dim=1)
    return pos_embed

def pi_resize_patch_embed(patch_embed: torch.Tensor, new_patch_size: Tuple[int, int], interpolation: str = "bicubic", antialias: bool = True):
    old_patch_size = tuple(patch_embed.shape[2:])
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x: torch.Tensor, shape: Tuple[int, int]):
        x_resized = F.interpolate(x[None, None, ...], shape, mode=interpolation, antialias=antialias)
        return x_resized[0, 0, ...]

    def calculate_pinv(old_shape: Tuple[int, int], new_shape: Tuple[int, int]):
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    pinv = calculate_pinv(old_patch_size, new_patch_size).to(patch_embed.device)

    def resample_patch_embed(patch_embed: torch.Tensor):
        h, w = new_patch_size
        resampled_kernel = pinv @ patch_embed.reshape(-1)
        return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

    v_resample_patch_embed = torch.vmap(torch.vmap(resample_patch_embed, 0, 0), 1, 1)

    return v_resample_patch_embed(patch_embed)

class FlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 256,
        patch_size: Union[int, Tuple[int, int]] = 32,
        in_chans: int = 3,
        embed_dim: int = 384,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
        bias: bool = True,
        patch_size_seq: Sequence[int] = (8, 10, 12, 15, 16, 20, 24, 30, 40, 48),
        patch_size_probs: Optional[Sequence[float]] = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.interpolation = interpolation
        self.antialias = antialias

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.flatten = flatten
        
        self.patch_size_seq = patch_size_seq
        if not patch_size_probs:
            n = len(self.patch_size_seq)
            self.patch_size_probs = [1.0 / n] * n
        else:
            self.patch_size_probs = [p / sum(patch_size_probs) for p in patch_size_probs]

    def forward(self, x: torch.Tensor, patch_size: Optional[Union[int, Tuple[int, int]]] = None):
        if not patch_size and not self.training:
            patch_size = self.patch_size
        elif not patch_size:
            patch_size = to_2tuple(np.random.choice(self.patch_size_seq, p=self.patch_size_probs))
            
        patch_size = to_2tuple(patch_size)  # Ensure patch_size is a tuple
        
        if patch_size != self.patch_size:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)
        else:
            weight = self.proj.weight

        x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)
        return x, patch_size

    def resize_patch_embed(self, patch_embed: torch.Tensor, new_patch_size: Tuple[int, int]):
        return pi_resize_patch_embed(
            patch_embed, new_patch_size, interpolation=self.interpolation, antialias=self.antialias
        )

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

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        
        # Apply self-attention with the reshaped attention mask
        x, _ = self.self_attention(x, x, x, attn_mask=attention_mask, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
    
class Encoder(nn.Module):
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

    def forward(self, input: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.dropout(input)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.ln(x)

def create_fractal_attention_mask(patch_size, img_size):
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]  # Use the first element if it's a tuple
    
    num_patches = (img_size // patch_size) ** 2
    mask = torch.ones(num_patches, num_patches)
    
    def recursive_mask(size, start_i, start_j):
        if size == 1:
            return
        half = size // 2
        mask[start_i:start_i+half, start_j+half:start_j+size] = 0
        mask[start_i+half:start_i+size, start_j:start_j+half] = 0
        recursive_mask(half, start_i, start_j)
        recursive_mask(half, start_i+half, start_j+half)
    
    recursive_mask(img_size // patch_size, 0, 0)
    return mask

class FlexibleVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        num_registers: int = 16,
        patch_size_seq: Sequence[int] = (8, 10, 12, 15, 16, 20, 24, 30, 40, 48),
        patch_size_probs: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_registers = num_registers
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_heads = num_heads  
        
        self.patch_embed = FlexiPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            patch_size_seq=patch_size_seq, patch_size_probs=patch_size_probs
        )
        num_patches = (img_size // patch_size) ** 2

        self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, embed_dim))
        self.global_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_registers + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.encoder = Encoder(
            seq_length=num_patches + num_registers + 1,
            num_layers=depth,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout=drop_rate,
            attention_dropout=attn_drop_rate,
            norm_layer=norm_layer,
        )
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self):
        nn.init.normal_(self.patch_embed.proj.weight, std=math.sqrt(2.0 / self.patch_embed.proj.out_channels))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        nn.init.normal_(self.register_tokens, std=.02)
        nn.init.normal_(self.global_token, std=.02)
        self.apply(self._init_weights_recursive)

    def _init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor, patch_size: Optional[Union[int, Tuple[int, int]]] = None):
        
        B = x.shape[0]
        x, patch_size = self.patch_embed(x, patch_size)

        register_tokens = self.register_tokens.expand(B, -1, -1)
        global_token = self.global_token.expand(B, -1, -1)
        x = torch.cat((x, register_tokens, global_token), dim=1)
    
        # Calculate the sequence length
        seq_length = x.size(1)
    
        attention_mask = create_fractal_attention_mask(patch_size, self.img_size)
        attention_mask = F.pad(attention_mask, (1 + self.num_registers, 0, 1 + self.num_registers, 0), value=1)
        attention_mask = -10000.0 * (1.0 - attention_mask)
    
        # Move attention_mask to the same device as x
        attention_mask = attention_mask.to(x.device)
    
        # Expand attention_mask for multi-head attention
        
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.expand(B, self.num_heads, seq_length, seq_length)
        attention_mask = attention_mask.reshape(B * self.num_heads, seq_length, seq_length)
        
        
        new_size = (self.img_size // patch_size[0], self.img_size // patch_size[1])
        
        # Resize positional embedding to match the new number of patches
        pos_embed = resize_abs_pos_embed(self.pos_embed, new_size, num_prefix_tokens=self.num_registers + 1)
    
        # Ensure pos_embed has the correct size
        if pos_embed.size(1) != x.size(1):
            raise ValueError(f"Positional embedding size {pos_embed.size(1)} does not match input size {x.size(1)}")

        x = self.pos_drop(x + pos_embed)

        x = self.encoder(x, attention_mask=attention_mask)
        x = self.norm(x)
        return x[:, -1]

    def forward(self, x, patch_size=None):
        x = self.forward_features(x, patch_size)
        x = self.head(x)
        return x

model = FlexibleVisionTransformer(
    img_size=256,
    patch_size=32,
    num_classes=100,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4,
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

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        batch_size = target.size(0)
        maxk = max(topk)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        if target.dim() > 1:  # mixed-up labels
            _, target = target.max(1)
        
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

log_steps = 2500

wandb.login(key="cbecbe8646ebcf42a98992be9fd5b7cddae3d199")

# Initialize a new run
wandb.init(project="fractual_transformer", name="More correct FlexVIT with mask attention only")

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
        
        if step % 38000 == 0 and step > 0:
            break

train(train_loader, val_loader, start_step, total_steps, original_model, model, criterion, optimizer, scheduler, device)

wandb.finish()
