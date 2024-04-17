# Prunability

## Torchvision 0.13.1
```python
python torchvision_pruning.py
```

#### Outputs:
```
Successful Pruning: 73 Models
 ['alexnet', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'googlenet', 'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf', 'regnet_y_128gf', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'ssdlite320_mobilenet_v3_large', 'ssd300_vgg16', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn']
```

```
Unsuccessful Pruning: 12 Models
 ['fcos_resnet50_fpn', 'raft_large', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'swin_t', 'swin_s', 'swin_b', 'keypointrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 'retinanet_resnet50_fpn_v2']
```

## YOLO v7

The following scripts (adapted from [yolov7/detect.py](https://github.com/WongKinYiu/yolov7/blob/main/detect.py) and [yolov7/train.py](https://github.com/WongKinYiu/yolov7/blob/main/train.py)) provides the basic examples of pruning YOLOv7. The training part has not been validated yet because it is quite time-consuming.

Note: [yolov7_detect_pruned.py](https://github.com/VainF/Torch-Pruning/blob/master/benchmarks/prunability/yolov7_detect_pruned.py) does not include any code for fine-tuning. 

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cp yolov7_detect_pruned.py yolov7/
cp yolov7_train_pruned.py yolov7/
cd yolov7 

# Test only: We only prune and test the YOLOv7 model in this script. COCO dataset is not required.
python yolov7_detect_pruned.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg

# Training with pruned yolov7 (The training part is not validated)
# Please download the pretrained yolov7_training.pt from https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt.
python yolov7_train_pruned.py --workers 8 --device 0 --batch-size 1 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml
```

Outputs of yolov7_detect_pruned.py:
```
Model(
  (model): Sequential(
    (0): Conv(
      (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
...
    (104): RepConv(
      (act): SiLU(inplace=True)
      (rbr_reparam): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (105): Detect(
      (m): ModuleList(
        (0): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(1024, 255, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)


Model(
  (model): Sequential(
    (0): Conv(
      (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
...
    (104): RepConv(
      (act): SiLU(inplace=True)
      (rbr_reparam): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (105): Detect(
      (m): ModuleList(
        (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)
Before Pruning: MACs=6.413721 G, #Params=0.036905 G
After Pruning: MACs=1.639895 G, #Params=0.009347 G
```

