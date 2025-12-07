#!/usr/bin/env python
"""Copyright(c) 2024. ProSe Training Script for Data-Incremental Object Detection
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import YAMLConfig
from data import build_dataloader
from nn.criterion import build_criterion
from zoo.rtdetr import RTDETR
from zoo.rtdetr.prose_rtdetrv2 import ProSeRTDETRv2


logger = logging.getLogger(__name__)


def setup_logging(rank, log_file=None):
    """Setup logging configuration"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(file_handler)


def build_model(cfg):
    """Build ProSe-RTDETRv2 model"""
    if cfg.get('prose', {}).get('use_prose', False):
        model = ProSeRTDETRv2(
            backbone=cfg.backbone,
            encoder=cfg.encoder,
            decoder=cfg.decoder,
            hidden_dim=cfg.prose.get('hidden_dim', 256),
            num_prototypes=cfg.prose.get('num_prototypes', 1200),
            num_heads=cfg.prose.get('num_heads', 8),
            alpha=cfg.prose.get('alpha', 1.0),
            lambda_weight=cfg.prose.get('lambda_weight', 0.5),
            use_gumbel_softmax=cfg.prose.get('use_gumbel_softmax', True),
            use_prose=True,
        )
        logger.info("Using ProSe-RTDETRv2 model")
    else:
        model = RTDETR(
            backbone=cfg.backbone,
            encoder=cfg.encoder,
            decoder=cfg.decoder,
        )
        logger.info("Using standard RT-DETRv2 model")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, rank, epoch, cfg):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images, targets)
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0 and (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
    
    if rank == 0:
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
    
    return total_loss / num_batches


def validate(model, dataloader, device, rank):
    """Validate model"""
    model.eval()
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
    
    if rank == 0:
        logger.info("Validation completed")


def main():
    parser = argparse.ArgumentParser(description='ProSe Training Script')
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file path')
    parser.add_argument('-r', '--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--increment-idx', type=int, default=0, help='Increment index for multi-increment training')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Setup logging
    setup_logging(rank, log_file=f'train_rank{rank}.log' if rank == 0 else None)
    
    # Load config
    cfg = YAMLConfig(args.config)
    
    if rank == 0:
        logger.info(f"Config loaded from {args.config}")
        logger.info(f"Training on {world_size} GPUs")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Build model
    model = build_model(cfg)
    model = model.to(device)
    
    # Wrap with DDP if distributed
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # Build dataloader
    train_loader = build_dataloader(
        cfg.data_cfg,
        batch_size=cfg.train_cfg.batch_size,
        num_workers=cfg.train_cfg.num_workers,
        is_train=True,
        rank=rank,
        world_size=world_size,
    )
    
    val_loader = build_dataloader(
        cfg.data_cfg,
        batch_size=cfg.train_cfg.batch_size,
        num_workers=cfg.train_cfg.num_workers,
        is_train=False,
        rank=rank,
        world_size=world_size,
    )
    
    # Build criterion
    criterion = build_criterion(cfg)
    
    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train_cfg.optimizer.lr,
        weight_decay=cfg.train_cfg.optimizer.weight_decay,
    )
    
    # Build scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.train_cfg.lr_scheduler.milestones,
        gamma=cfg.train_cfg.lr_scheduler.gamma,
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        if rank == 0:
            logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    num_epochs = cfg.train_cfg.epochs
    
    for epoch in range(start_epoch, num_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, rank, epoch, cfg)
        
        # Validate
        if (epoch + 1) % cfg.eval_cfg.eval_interval == 0:
            validate(model, val_loader, device, rank)
        
        # Save checkpoint
        if rank == 0 and (epoch + 1) % cfg.eval_cfg.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
            }
            save_path = f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save(checkpoint, save_path)
            logger.info(f"Checkpoint saved to {save_path}")
        
        # Update learning rate
        scheduler.step()
    
    if rank == 0:
        logger.info("Training completed!")


if __name__ == '__main__':
    main()
