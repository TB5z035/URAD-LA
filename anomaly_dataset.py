from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os
import os.path as osp
from PIL import Image
from config import cfg
from utils.misc import print0

ds_path = cfg.ANOMALY_DATASET_DIR
ds_path.mkdir(exist_ok=True)

def gen_fslaf():
    print("Loading Fishyscapes LostAndFound images and masks")
    import bdlb
    fs = bdlb.load(benchmark="fishyscapes", download_and_prepare=False)
    fs.download_and_prepare('LostAndFound')

    import tensorflow_datasets as tfds

    ds = tfds.load('fishyscapes/LostAndFound', split='validation')

    basedata_id_list = []
    image_id_list = []
    image_list = []
    mask_list = []
    for i, blob in enumerate(tqdm(ds.take(100))):
        basedata_id = blob['basedata_id'].numpy()
        image_id = blob['image_id'].numpy()
        image = blob['image_left'].numpy()
        mask = blob['mask'].numpy()
        mask[mask == 255] = 2

        basedata_id_list.append(basedata_id)
        image_id_list.append(image_id)
        image_list.append(image)
        mask_list.append(mask[..., 0])
    np.savez(ds_path / 'fslaf.npz', np.array(image_list), np.array(mask_list))

def gen_ra():
    print("Loading RoadAnomaly images and masks")
    root = "RoadAnomaly/frames"
    images = []
    labels = []
    for f in tqdm(sorted(os.listdir(root))):
        fpath = osp.join(root, f)
        name, ext = osp.splitext(f)
        if osp.isfile(fpath) and ext == '.webp':
            images += [np.asarray(Image.open(fpath))]
            labels += [np.asarray(Image.open(osp.join(root, name + ".labels", "labels_semantic.png")))]

    for mask in labels:
        ood = mask == 2
        ind = mask == 0
        assert (ood | ind).all()
        mask[ood] = 1
        mask[ind] = 0
    np.savez(ds_path / "ra.npz", np.array(images), np.array(labels))

def gen_laf():
    print("Loading LostAndFound images and masks")
    import tensorflow_datasets as tfds
    dsg = tfds.image.lost_and_found.LostAndFound()
    dsg.download_and_prepare()
    ds = dsg.as_dataset('test', shuffle_files=False)

    image_list = []
    mask_list = []  
    for blob in tqdm(ds.take(1203)):
        image = blob['image_left'].numpy()
        mask = blob['segmentation_label'].numpy()

        doc = mask == 0 
        ind = mask == 1
        ood = mask > 1

        mask[ind] = 0
        mask[ood] = 1
        mask[doc] = 2
        
        image_list.append(image)
        mask_list.append(mask[..., 0])
    np.savez(ds_path / 'laf.npz', np.array(image_list), np.array(mask_list))

def get_anomaly_dataset(ds_name):
    print0(f"[Loading {ds_name} ...]")
    ds = np.load(ds_path / (ds_name + ".npz"))
    image_list = ds['arr_0']
    mask_list = ds['arr_1']
    return image_list, mask_list

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    parser = ArgumentParser()
    parser.add_argument('--prepare', type=str, nargs='+', choices=['fslaf', 'laf', 'ra'])
    args = parser.parse_args()

    if len(args.prepare):
        ds_path.mkdir(exist_ok=True)

    if 'fslaf' in args.prepare:
        gen_fslaf()

    if 'laf' in args.prepare:
        gen_laf()

    if 'ra' in args.prepare:
        gen_ra()