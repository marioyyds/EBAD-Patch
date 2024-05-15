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
# target_label_set = set([0, 10, 2, 5, 6, 7, 1, 3])
target_label_set = set([0, 10, 2, 5, 6, 7, 1, 3, 9, 11])

def patch_initialization(x_width, y_width):
    # patch = np.random.randint(0, 256, size=(3, x_width, y_width), dtype=np.uint8)
    patch = np.load("patch/patch3.npy")
    # patch[patch < 20] = 0
    return patch

def patch_mask_generation(patch = None,image_size=(3, 224, 224), bounding_boxes = None):
    applied_patch = np.zeros(image_size)
    applied_patch_loc = []

    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        x_width = int(x2) - int(x1)
        y_width = int(y2) - int(y1)

        if x_width > patch.shape[1] and y_width > patch.shape[2]:
            for mask_num in range(3):
                # for mask_num in range(2):
                #     # patch location
                x_location, y_location = np.random.randint(low=0, high=x_width-patch.shape[1]), np.random.randint(low=0, high=y_width-patch.shape[2])
                
                applied_patch[:, int(y1) + y_location:int(y1) + y_location + patch.shape[2], int(x1) + x_location:int(x1) + x_location + patch.shape[1]] = patch
                applied_patch_loc.append((int(x1) + x_location, int(y1) + y_location))
        elif int(x1) + patch.shape[1] < image_size[1] and int(y1) + patch.shape[2] < image_size[2]:
            x_location, y_location = 0, 0
            
            applied_patch[:, int(y1) + y_location:int(y1) + y_location + patch.shape[2], int(x1) + x_location:int(x1) + x_location + patch.shape[1]] = patch
            applied_patch_loc.append((int(x1) + x_location, int(y1) + y_location))

    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    mask = np.logical_not(mask)
    return applied_patch, applied_patch_loc, mask

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
    im_path = f"/data/hdd3/duhao/code/EBAD1/data/29/{img}.jpg"
    ori_im_path = f"/data/hdd3/duhao/code/EBAD1/data/29/{img}.jpg"
    adv_path = f"/data/hdd3/duhao/code/EBAD1/mask_new_all_4/{img}.jpg"
    im_np = np.array(Image.open(im_path).convert('RGB'))
    ori_im_np = np.array(Image.open(ori_im_path).convert('RGB'))

    # get detection on clean images and determine target class
    det = model.det(ori_im_np)

    indices_to_remove = np.any(det[:, 4:5] == np.array(list(target_label_set)), axis=1)
    det = det[indices_to_remove]

    bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]


    im_np = im_np.transpose(2, 0, 1)
    patch = patch_initialization(0, 0)
    applied_patch, applied_patch_loc, mask = patch_mask_generation(patch, im_np.shape, bboxes)

    mask_not = np.logical_not(mask)
    mask_new = im_np*mask + applied_patch * mask_not

    mask_new_im = mask_new.transpose(1, 2, 0).astype(np.uint8)

    mask_new_im.clip(0, 255)
    # mask_im.clip(0, 255)

    adv_png = Image.fromarray(mask_new_im.astype(np.uint8))
    adv_png.save(adv_path)
    print("sucessfully saved: ", img)
