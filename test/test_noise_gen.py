import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import numpy as np
from utils_mmdet import model_train
from PIL import Image
import matplotlib.pyplot as plt 
import json as JSON

target_label_set = set([0, 2, 3, 9, 11])
# target_label_set = set([0, 10, 2, 5, 6, 7, 1, 3, 9, 11])
# target_label_set = set([0, 10, 2, 5, 6, 7, 1, 3])

def generate_mask(image_shape, bounding_boxes):
    mask = np.zeros(image_shape, dtype=np.uint8)

    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
        # break
    return mask

def generate_noise(image_shape, bounding_boxes):
    # noise = np.zeros(image_shape, dtype=np.uint8)

    low, high = 50, 51  # 范围为0到10

    noise = torch.randint(low, high, image_shape)
        # break
    return noise

model = model_train(model_name="DETR", dataset="coco")

test_image_ids = JSON.load(open(f"data/CVPR_Adversarial/output.json"))

for img in test_image_ids:
    im_path = f"/data/hdd3/duhao/code/EBAD1/data/CVPR_Adversarial/{img}.jpg"
    adv_path = f"/data/hdd3/duhao/code/EBAD1/noise_all/{img}.jpg"
    im_np = np.array(Image.open(im_path).convert('RGB'))

    # get detection on clean images and determine target class
    det = model.det(im_np)

    indices_to_remove = np.any(det[:, 4:5] == np.array(list(target_label_set)), axis=1)
    det = det[indices_to_remove]

    bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]

    # im = torch.from_numpy(im_np).permute(2,0,1).unsqueeze(0).float().to('cuda')
    im = torch.from_numpy(im_np).permute(2,0,1).float()

    mask = generate_mask(im.shape[-2:], bboxes)

    noise = generate_noise(im.shape[-2:], bboxes)
    noise_mask = noise.masked_fill_(torch.logical_not(torch.from_numpy(mask)), 0)
    no_im = noise_mask + im

    # origin_im = im.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    mask_im = no_im.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

    mask_im.clip(0, 255)

    adv_png = Image.fromarray(mask_im.astype(np.uint8))
    adv_png.save(adv_path)
