OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=8 --master_port 18113 --use_env main_imagenet.py --model resnet50 --epochs 90 --batch-size 256 --lr 0.08 --prune --cache-dataset --method group_norm --soft-keeping-ratio 0.5 --pretrained --output-dir run/imagenet/resnet50_gnorm --target-flops 2.04 --global-pruning --print-freq 100 --workers 8