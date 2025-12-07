#!/usr/bin/env python
"""Copyright(c) 2024. ProSe Inference Script for Data-Incremental Object Detection
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import YAMLConfig
from zoo.rtdetr.prose_rtdetrv2 import ProSeRTDETRv2


class ProSeInference:
    """Inference wrapper for ProSe-RTDETRv2"""
    
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        """Initialize inference engine
        
        Args:
            config_path: path to config file
            checkpoint_path: path to checkpoint file
            device: device to run inference on
        """
        self.device = torch.device(device)
        
        # Load config
        self.cfg = YAMLConfig(config_path)
        
        # Build model
        self.model = ProSeRTDETRv2(
            backbone=self.cfg.backbone,
            encoder=self.cfg.encoder,
            decoder=self.cfg.decoder,
            hidden_dim=self.cfg.prose.get('hidden_dim', 256),
            num_prototypes=self.cfg.prose.get('num_prototypes', 1200),
            num_heads=self.cfg.prose.get('num_heads', 8),
            alpha=self.cfg.prose.get('alpha', 1.0),
            lambda_weight=self.cfg.prose.get('lambda_weight', 0.5),
            use_gumbel_softmax=self.cfg.prose.get('use_gumbel_softmax', True),
            use_prose=self.cfg.prose.get('use_prose', True),
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Class names (COCO)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
            'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def preprocess(self, image_path, img_size=640):
        """Preprocess image
        
        Args:
            image_path: path to image file
            img_size: target image size
        
        Returns:
            image_tensor: preprocessed image tensor [1, 3, H, W]
            original_size: original image size (H, W)
        """
        # Read image
        image = cv2.imread(image_path)
        original_size = image.shape[:2]
        
        # Resize
        h, w = image.shape[:2]
        scale = img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        pad_h = img_size - new_h
        pad_w = img_size - new_w
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert BGR to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor [3, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        # Add batch dimension [1, 3, H, W]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, original_size
    
    def postprocess(self, outputs, original_size, conf_threshold=0.5, nms_threshold=0.5):
        """Postprocess model outputs
        
        Args:
            outputs: model outputs
            original_size: original image size (H, W)
            conf_threshold: confidence threshold
            nms_threshold: NMS threshold
        
        Returns:
            detections: list of detections with format [x1, y1, x2, y2, conf, class_id]
        """
        pred_logits = outputs['pred_logits']  # [1, 300, 80]
        pred_boxes = outputs['pred_boxes']    # [1, 300, 4]
        
        # Get confidence scores
        scores = torch.sigmoid(pred_logits[0])  # [300, 80]
        
        # Get class predictions
        class_ids = torch.argmax(scores, dim=-1)  # [300]
        confidences = scores[torch.arange(len(scores)), class_ids]  # [300]
        
        # Filter by confidence threshold
        mask = confidences > conf_threshold
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        boxes = pred_boxes[0][mask]  # [N, 4] in cxcywh format
        
        if len(boxes) == 0:
            return []
        
        # Convert boxes from cxcywh to xyxy
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        
        # Denormalize boxes to original image size
        h_orig, w_orig = original_size
        boxes = boxes * torch.tensor([w_orig, h_orig, w_orig, h_orig], device=boxes.device)
        
        # NMS
        keep = self._nms(boxes, confidences, nms_threshold)
        
        detections = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx].cpu().numpy()
            conf = confidences[idx].item()
            class_id = class_ids[idx].item()
            detections.append([x1, y1, x2, y2, conf, class_id])
        
        return detections
    
    def _nms(self, boxes, scores, threshold=0.5):
        """Non-maximum suppression"""
        x1, y1, x2, y2 = boxes.unbind(-1)
        areas = (x2 - x1) * (y2 - y1)
        
        _, order = scores.sort(descending=True)
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Compute IoU with remaining boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / union
            
            # Keep boxes with IoU below threshold
            mask = iou <= threshold
            order = order[1:][mask]
        
        return torch.tensor(keep, device=boxes.device)
    
    def visualize(self, image_path, detections, output_path=None):
        """Visualize detections
        
        Args:
            image_path: path to input image
            detections: list of detections
            output_path: path to save visualization (optional)
        """
        image = cv2.imread(image_path)
        
        for x1, y1, x2, y2, conf, class_id in detections:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{self.class_names[int(class_id)]}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
    
    def infer(self, image_path, conf_threshold=0.5, return_branch_idx=False):
        """Run inference on image
        
        Args:
            image_path: path to image file
            conf_threshold: confidence threshold
            return_branch_idx: whether to return selected branch index
        
        Returns:
            detections: list of detections
            branch_idx: selected branch index (if return_branch_idx=True)
        """
        # Preprocess
        image_tensor, original_size = self.preprocess(image_path)
        
        # Inference
        with torch.no_grad():
            if return_branch_idx:
                outputs, branch_idx = self.model(image_tensor, return_branch_idx=True)
            else:
                outputs = self.model(image_tensor)
                branch_idx = None
        
        # Postprocess
        detections = self.postprocess(outputs, original_size, conf_threshold)
        
        if return_branch_idx:
            return detections, branch_idx
        else:
            return detections


def main():
    parser = argparse.ArgumentParser(description='ProSe Inference Script')
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file path')
    parser.add_argument('-r', '--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('-i', '--image', type=str, required=True, help='Image path or directory')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output directory')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = ProSeInference(args.config, args.checkpoint, device=args.device)
    
    # Create output directory if needed
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Process images
    if os.path.isdir(args.image):
        image_files = [f for f in os.listdir(args.image) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        image_files = [os.path.basename(args.image)]
        args.image = os.path.dirname(args.image)
    
    for image_file in image_files:
        image_path = os.path.join(args.image, image_file)
        
        # Run inference
        detections, branch_idx = inference.infer(image_path, args.conf_threshold, return_branch_idx=True)
        
        print(f"Image: {image_file}")
        print(f"Selected branch: {branch_idx}")
        print(f"Detections: {len(detections)}")
        for x1, y1, x2, y2, conf, class_id in detections:
            print(f"  {inference.class_names[int(class_id)]}: {conf:.3f}")
        
        # Visualize
        if args.output:
            output_path = os.path.join(args.output, f"result_{image_file}")
            inference.visualize(image_path, detections, output_path)
            print(f"Saved to {output_path}")
        print()


if __name__ == '__main__':
    main()
