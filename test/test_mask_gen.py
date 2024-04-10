import sys 
sys.path.append(__file__.rsplit('/', 2)[0])
sys.path.append("..")
print(sys.path)

import torch
import numpy as np
from utils_mmdet import model_train
from PIL import Image
import matplotlib.pyplot as plt 

target_label_set = set([0, 2, 3, 9, 11])

def generate_mask(image_shape, bounding_boxes):
    mask = np.zeros(image_shape, dtype=np.uint8)

    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
    return mask

model = model_train(model_name="DETR", dataset="coco")

im_path = "/data/hdd3/duhao/code/EBAD/data/CVPR_Adversarial/000004.jpg"
im_np = np.array(Image.open(im_path).convert('RGB'))

# get detection on clean images and determine target class
det = model.det(im_np)

indices_to_remove = np.any(det[:, 4:5] == np.array(list(target_label_set)), axis=1)
det = det[indices_to_remove]

bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]

# im = torch.from_numpy(im_np).permute(2,0,1).unsqueeze(0).float().to('cuda')
im = torch.from_numpy(im_np).permute(2,0,1).float()

mask = generate_mask(im.shape[-2:], bboxes)

image_mask = im.masked_fill(torch.from_numpy(mask), 0)

origin_im = im.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
mask_im = image_mask.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 2)
ax.imshow(mask_im)

ax = fig.add_subplot(1, 2, 1)
ax.imshow(origin_im)

plt.savefig("gen_mask.jpg")

