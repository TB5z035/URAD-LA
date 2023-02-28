# Unsupervised Road Anomaly Detection with Language Anchors

Offical implementation of *Unsupervised Road Anomaly Detection with Language Anchors* (ICRA 2023)

## Prepreration

### Configure Environment

Run
```bash
./scripts/install.sh
```

### Semantic Segmentation Dataset

Arange *Cityscapes* dataset like

```
./cityscapes/
├── gtFine_trainvaltest
│   └── gtFine
│       ├── test
│       ├── train
│       └── val
└── leftImg8bit_trainvaltest
    └── leftImg8bit
        ├── test
        ├── train
        └── val
```

### Anomaly Detection Dataset

We use Fishyscapes LostAndFound, LostAndFound and RoadAnomaly as measurement. Steps to prepare them:

- *Fishyscapes LostAndFound* and *LostAndFound* can be obtained from TensorFlow dataset, so we don't have to download them manually.

- Download [RoadAnomaly Dataset](https://www.epfl.ch/labs/cvlab/data/road-anomaly/) from https://datasets-cvlab.epfl.ch/2019-road-anomaly/RoadAnomaly.zip and unzip it like
    ```
    ./RoadAnomaly
    ├── frames
    │   ├── animals01_Guiguinto_railway_station_Calves.labels
    │   ├── animals01_Guiguinto_railway_station_Calves.webp
    │   └── ...
    └── ...
    ```

- Run 
    ```bash
    python ./anomaly_dataset.py --prepare fslaf laf ra
    ```
    which will convert anomaly datasets into `npz` files, which will be used for evaluation

## Training and Evaluation

### Training

To train the network, run

```bash
bash ./scripts/train.sh
```

Some options for training in the script:
- `--disable_le`: when this flag is set, the model will not use language anchor
- `--logit_type`: `anchor` and `binary` for anchor logit and binary logit
- `--T`, `--tau`, `--inf_temp` temperature arguments for log softmax, cross entropy loss and inference(BSL).


### Evalutation

To evalute the anomaly detection performance of a checkpoint, run
```bash
bash ./scripts/evaluate.sh
```

Some options for evalutation in the script:
- `--snapshot`: path to the checkpoint
- `--score_mode`: anomaly score mode to use, can be `bsl`, `sml`, `ml`, `msp`, `entropy`.
- `--inference_scale`: inference scales for multi-scale inference
- `--anomaly_dataset`: choose the anomaly detection dataset to evaluate the performance

More training and evaluation options see `options.py`.

## Pretrained models

Download pretrained models from [[Google Drive]](https://drive.google.com/drive/folders/1ZTawZ-MePaMQW8MdIKXfyVcAPZUqDBNM?usp=sharing). Place `pth` file in `./pretrained` directory.

## Acknowledgement

Our code is derived from [SML](https://github.com/shjung13/Standardized-max-logits). Thanks to their great work and open source repository.