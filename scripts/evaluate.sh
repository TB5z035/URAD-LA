#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,2,3,6
NGPU=$((`echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c` + 1))

# evaluate results on Fishyscapes LostAndFound
python -m torch.distributed.launch --nproc_per_node=$NGPU evaluate.py \
    --score_mode bsl \
    --snapshot pretrained/fslaf.pth \
    --inference_scale 0.5 0.65 0.85 1.0 1.25 1.75 \
    --inf_temp 1.0 \
    --anomaly_dataset fslaf


# evaluate results on RoadAnomaly
python -m torch.distributed.launch --nproc_per_node=$NGPU evaluate.py \
    --score_mode bsl \
    --snapshot ./pretrained/ra.pth \
    --inference_scale 0.5 0.65 0.85 1.0 1.25 1.75 \
    --inf_temp 1.0 \
    --anomaly_dataset ra


# evaluate results on LostAndFound
python -m torch.distributed.launch --nproc_per_node=$NGPU evaluate.py \
    --score_mode bsl \
    --snapshot pretrained/laf.pth \
    --inference_scale 0.5 0.65 0.85 1.0 1.25 1.75 \
    --inf_temp 1.0 \
    --anomaly_dataset laf
