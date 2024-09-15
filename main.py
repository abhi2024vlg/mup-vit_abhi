#This is BaseLine_VIT_more_fractal

### Necessary Imports and dependencies

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
from torchvision.transforms import v2
import torchvision.transforms as transforms
from typing import Any, Dict, Union, Type, Callable, Optional, List
from torchvision.models.vision_transformer import MLPBlock
import wandb
import json
from PIL import Image
from torch.utils.data import Dataset

#No. of Epochs
num_epochs=30

#Warmp_Steps
warmup_try=1000

# Parameters specific to ImageNet100
batch_size = 256

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


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

def generate_combined_posemb(dim, device):
    # Generate embeddings for each part
    pe_256 = posemb_sincos_2d(16, 16, dim).to(device)
    pe_16 = posemb_sincos_2d(4, 4, dim).to(device)
    pe_1 = posemb_sincos_2d(1, 1, dim).to(device)

    # Concatenate along the sequence dimension
    combined_pe = torch.cat([pe_256, pe_16, pe_1], dim=0)
    return combined_pe

def jax_lecun_normal(layer, fan_in):
    """(re-)initializes layer weight in the same way as jax.nn.initializers.lecun_normal and bias to zero"""

    # constant is stddev of standard normal truncated to (-2, 2)
    std = math.sqrt(1 / fan_in) / .87962566103423978
    nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
        
def create_fractal_attention_mask(n_h, n_w):
    
    base_len = n_h * n_w
    # Create mask for 16x16 grid: Tokens can only attend to their 4x4 patch mates.
    mask_16x16 = torch.zeros(base_len, base_len)
    for i in range(n_h // 4):
        for j in range(n_w // 4):
            for r1 in range(i * 4, i * 4 + 4):
                s1 = r1 * n_w + j * 4
                for r2 in range(i * 4, i * 4 + 4):
                    s2 = r2 * n_w + j * 4
                    mask_16x16[s1:s1 + 4, s2:s2 + 4] = 1
    
    # Create mask for 4x4 summary tokens
    mask_4x4 = torch.ones(base_len // 16, base_len // 16)
    
    # Create mask for global token
    mask_global = torch.ones(1, 1)
    
    # Combine masks
    mask = torch.block_diag(mask_16x16, mask_4x4, mask_global)
    
    # Allow 4x4 summary tokens to attend to their corresponding 4x4 regions, and vice versa.
    for i in range(n_h // 4):
        for j in range(n_w // 4):
            index = base_len + i * 4 + j
            for row in range(i * 4, i * 4 + 4):
                start = row * n_w + j * 4
                mask[start:start + 4, index] = mask[index, start:start + 4] = 1
    
    # Allow global token to attend to summary tokens, and vice versa.
    
    mask[-1, base_len:base_len + base_len // 16] = 1
    mask[base_len:base_len + base_len // 16, -1] = 1
    
    return mask

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

        # Initialize attention layers
        # Fix init discrepancy between nn.MultiheadAttention and that of big_vision
        bound = math.sqrt(3 / hidden_dim)
        nn.init.uniform_(self.self_attention.in_proj_weight, -bound, bound)
        nn.init.uniform_(self.self_attention.out_proj.weight, -bound, bound)

    def forward(self, input: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        
        # Apply self-attention with fractal attention masking
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
            x = layer(x, attention_mask)
        return self.ln(x)

class SimpleVisionTransformer(nn.Module):
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
        self.num_heads = num_heads

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        grid_size = image_size // patch_size
        self.grid_size = grid_size
        seq_length = grid_size * grid_size + grid_size * grid_size // 16 + 1  # Image patches + summary tokens + global token
        self.seq_length = seq_length

        # Initialize position embeddings
        self.register_buffer("pos_embedding", generate_combined_posemb(hidden_dim, self.conv_proj.weight.device))

        # Create fractal attention mask
        self.register_buffer("attention_mask", create_fractal_attention_mask(grid_size, grid_size))

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
        # Init the patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1] // self.conv_proj.groups
        jax_lecun_normal(self.conv_proj, fan_in)
        
        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            jax_lecun_normal(self.heads.pre_logits, fan_in)
            
        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")

        # (n, c, h, w) -> (n, hidden_dim, grid_size, grid_size)
        x = self.conv_proj(x)
        
        # (n, hidden_dim, grid_size, grid_size) -> (n, grid_size * grid_size, hidden_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Add summary tokens (zero-initialized)
        summary_tokens = torch.zeros((n, self.grid_size * self.grid_size // 16, self.hidden_dim), device=x.device)
        
        # Add global token (zero-initialized)
        global_token = torch.zeros((n, 1, self.hidden_dim), device=x.device)
        
        # Combine all tokens
        x = torch.cat([x, summary_tokens, global_token], dim=1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Add positional encoding
        x = x + self.pos_embedding.unsqueeze(0).expand(n, -1, -1)

        # Prepare attention mask for the encoder
        attention_mask = self.attention_mask.unsqueeze(0).expand(n, -1, -1)
        attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attention_mask = attention_mask.reshape(n * self.num_heads, self.seq_length, self.seq_length)
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))

        # Apply transformer
        x = self.encoder(x, attention_mask=attention_mask)
        
        # Use the global token for classification
        x = x[:, -1]

        # Apply classification head
        x = self.heads(x)

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
)

model = nn.DataParallel(model)
model.to('cuda')

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
    filename = os.path.join(checkpoint_path, f'(Modified)BaseLine_VIT_more_fractal.pt')
    
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

wandb.login(key="cbecbe8646ebcf42a98992be9fd5b7cddae3d199")

# Initialize a new run
wandb.init(project="fractual_transformer", name="ImageNet100_Baseline(modified)_run_more_fractal_PE_1000")

def validate(val_loader, model, criterion, step, use_wandb=False, print_freq=100):
    
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

                # compute output
                output = model(images)
                loss = criterion(output,target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output,target, topk=(1, 5))
                losses.update(loss.item(),images.size(0))
                top1.update(acc1[0].item(),images.size(0))
                top5.update(acc5[0].item(),images.size(0))
                    
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
        
        output = model(images)
        loss = criterion(output,target)
        
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5), class_prob=True)
        step_loss += loss.item()
        step_acc1 += acc1[0].item()
        step_acc5 += acc5[0].item()
        
        losses.update(step_loss, images.size(0))
        top1.update(step_acc1, images.size(0))
        top5.update(step_acc5, images.size(0))
        
        # compute gradient and do SGD step
        loss.backward()
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
        
        if ((step % print_freq == 0) and ((step % log_steps != 0) and (step != total_steps))):        
            
            save_checkpoint_step(step, model, best_acc1, optimizer, scheduler, checkpoint_path)
                
        if step % log_steps == 0:
            
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
            
        elif step == total_steps:
            
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
        

train(train_loader,val_loader, start_step, total_steps, original_model, model, criterion, optimizer, scheduler, device)

wandb.finish()
