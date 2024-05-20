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

# target_label_set = set([0, 2, 3, 9, 11])
target_label_set = set([0, 10, 2, 5, 6, 7, 1, 3, 9, 11])
# target_label_set = set([0, 10, 2, 5, 6, 7, 1, 3])

def select_random_region(image, bbox, size):
    x1, y1, x2, y2 = bbox
    max_x = x2 - size[0]
    max_y = y2 - size[1]
    if max_x < x1 or max_y < y1:
        return None
    rand_x = np.random.randint(x1, max_x)
    rand_y = np.random.randint(y1, max_y)
    return rand_y,rand_x

def generate_mask_new(ori_img, image_shape, bounding_boxes):
    # mask = np.zeros(image_shape, dtype=np.uint8)
    mask = ori_img

    
    # random_mask = np.random.randint(0, 256, size=mask_size, dtype=np.uint8)  # 随机生成 mask


    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        number = int(((x2-x1)*(y2-y1))/100)
        number = min(number, 30)
        # print(number)
        for i in range(number):
            mask_size = (min(5,int((y2 - y1)/4)), min(5,int((x2 - x1)/4)))  # mask 大小
            # print(mask_size)
            rand_y,rand_x = select_random_region(mask, box, mask_size)
            # print(mask_size)
            for j in range(3):
                random_mask = np.random.randint(0, 256, size=mask_size, dtype=np.uint8)
                # print(random_mask.shape)
                mask[j, rand_y:(rand_y+mask_size[0]), rand_x:(rand_x+mask_size[1])] = torch.from_numpy(random_mask)
            # mask[int(y1):int(y2), int(x1):int(x2)] = random_mask
        # break
    # for box in bounding_boxes:
    #     x1, y1, x2, y2 = box
    #     for i in range(10):
    #         mask_size = (int((y2 - y1)/4), int((x2 - x1)/4))  # mask 大小
    #         rand_y,rand_x = select_random_region(mask, box, mask_size)
    #         # print(mask.shape)
    #         for j in range(3):
    #             random_mask = np.random.randint(0, 256, size=mask_size, dtype=np.uint8)
    #             mask[j, rand_y:(rand_y+mask_size[0]), rand_x:(rand_x+mask_size[0])] = torch.from_numpy(random_mask)
    #         # mask[int(y1):int(y2), int(x1):int(x2)] = random_mask
    #     # break
    return mask

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

model = model_train(model_name="CO-DETR", dataset="coco")

test_image_ids = JSON.load(open(f"data/test_phase2/output.json"))

for img in test_image_ids:
    im_path = f"/data/hdd3/duhao/code/EBAD1/data/test_phase2/{img}.jpg"
    im_ens_adv_path = f"/data/hdd3/duhao/code/CVPR_Workshop24_Adversarial/data/phase2_16/images/{img}.jpg"
    adv_path = f"/data/hdd3/duhao/code/EBAD1/mask_new_all_3/{img}.jpg"
    im_np = np.array(Image.open(im_path).convert('RGB'))
    im_ens_adv_np = np.array(Image.open(im_ens_adv_path).convert('RGB'))

    # get detection on clean images and determine target class
    det = model.det(im_np)

    indices_to_remove = np.any(det[:, 4:5] == np.array(list(target_label_set)), axis=1)
    det = det[indices_to_remove]

    bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]

    # im = torch.from_numpy(im_np).permute(2,0,1).unsqueeze(0).float().to('cuda')
    im = torch.from_numpy(im_ens_adv_np).permute(2,0,1).float()

    # mask = generate_mask(im.shape[-2:], bboxes)
    # noise = generate_noise(im.shape[-2:], bboxes)

    mask_new = generate_mask_new(im, im.shape[-2:], bboxes)

    # noise_mask = noise.masked_fill_(torch.logical_not(torch.from_numpy(mask)), 0)
    # no_im = noise_mask + im

    # origin_im = im.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    # mask_im = no_im.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    mask_new_im = mask_new.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

    mask_new_im.clip(0, 255)
    # mask_im.clip(0, 255)

    adv_png = Image.fromarray(mask_new_im.astype(np.uint8))
    adv_png.save(adv_path)
    print("sucessfully saved: ", img)
