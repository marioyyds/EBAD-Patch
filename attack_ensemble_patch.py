import sys
import argparse
from pathlib import Path
from collections import defaultdict
import json as JSON
import random
import pdb

import datetime

import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm



mmdet_root = Path('mmdetection/')
sys.path.insert(0, str(mmdet_root))
from utils_mmdet import vis_bbox, VOC_BBOX_LABEL_NAMES, COCO_BBOX_LABEL_NAMES, voc2coco, get_det, is_success, get_iou
from utils_mmdet import model_train

target_label_set = set([0, 10, 2, 5, 6, 7, 1, 3, 9, 11])

def patch_initialization(image_size=(3, 224, 224), noise_percentage=0.06):

    mask_length = 40
    patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

def patch_mask_generation(patch=None, image_size=(3, 224, 224), bounding_boxes = None):
    applied_patch = np.zeros(image_size)
    applied_patch_loc = []

    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        x_width = int(x2) - int(x1)
        y_width = int(y2) - int(y1)

        if patch.shape[1] < x_width*0.75 and patch.shape[2] < y_width*0.75:
            for mask_num in range(2):
                # patch location
                x_location, y_location = np.random.randint(low=0, high=x_width-patch.shape[1]), np.random.randint(low=0, high=y_width-patch.shape[2])
                
                applied_patch[:, int(y1) + y_location:int(y1) + y_location + patch.shape[2], int(x1) + x_location:int(x1) + x_location + patch.shape[1]] = patch
                applied_patch_loc.append((int(x1) + x_location, int(y1) + y_location))
        elif int(x1) + x_width/2 + patch.shape[1] < image_size[1] and int(y1) + y_width/2 + patch.shape[2] < image_size[2]:
            x_location, y_location = int(x_width/2), int(y_width/2)
            
            applied_patch[:, int(y1) + y_location:int(y1) + y_location + patch.shape[2], int(x1) + x_location:int(x1) + x_location + patch.shape[1]] = patch
            applied_patch_loc.append((int(x1) + x_location, int(y1) + y_location))

    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    mask = np.logical_not(mask)
    return applied_patch, applied_patch_loc, mask

def generate_mask(image_shape, bounding_boxes):
    mask = np.ones(image_shape, dtype=np.uint8)

    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        
        mask[int(y1):int(y2), int(x1):int(x2)] = 0
    return mask

def PM_tensor_weight_balancing(im, adv, target, w, ensemble, eps, n_iters, alpha, dataset='voc', weight_balancing=False, patch = None):
    """perturbation machine, balance the weights of different surrogate models
    args:
        im (tensor): original image, shape [1,3,h,w].cuda()
        adv (tensor): adversarial image
        target (numpy.ndarray): label for object detection, (xyxy, cls, conf)
        w (numpy.ndarray): ensemble weights
        ensemble (): surrogate ensemble
        eps (int): linf norm bound (0-255)
        n_iters (int): number of iterations
        alpha (flaot): step size

    returns:
        adv_list (list of Tensors): list of adversarial images for all iterations
        LOSS (dict of lists): 'ens' is the ensemble loss, and other individual surrogate losses
    """
    # prepare target label input: voc -> coco, since models are trained on coco
    bboxes_tgt = target[:,:4].astype(np.float32)
    labels_tgt = target[:,4].astype(int).copy()
    if dataset == 'voc':
        for i in range(len(labels_tgt)): 
            labels_tgt[i] = voc2coco[labels_tgt[i]]

    im_np = im.squeeze().cpu().numpy().transpose(1, 2, 0)
    adv_list = []
    pert = adv - im
    LOSS = defaultdict(list) # loss lists for different models
    # im_np
    applied_patch, applied_patch_loc, patch_mask = patch_mask_generation(patch, im.shape[1:], bboxes_tgt)
    applied_patch = torch.from_numpy(applied_patch).unsqueeze(0).float()
    patch_mask = patch_mask.transpose(1, 2, 0)

    for i in range(n_iters):
        pert.requires_grad = True
        loss_list = []
        loss_list_np = []
        for model in ensemble:
            loss = model.loss(im_np, pert, bboxes_tgt, labels_tgt)
            loss_list.append(loss)
            loss_list_np.append(loss.item())
            LOSS[model.model_name].append(loss.item())
        
        # if balance the weights at every iteration
        if weight_balancing:
            w_inv = 1/np.array(loss_list_np)
            w = w_inv / w_inv.sum()

        # print(f"w: {w}")
        loss_ens = sum(w[i]*loss_list[i] for i in range(len(ensemble)))
        loss_ens.backward()
        with torch.no_grad():
            pert = pert - alpha*torch.sign(pert.grad)
            pert = pert.clamp(min=-eps, max=eps)
            LOSS['ens'].append(loss_ens.item())
      
            # add mask to attack only specify objection area/range
            mask = torch.from_numpy(generate_mask(pert.shape[-2:], bboxes_tgt)).to(pert.device)
            pert = pert.masked_fill(mask.bool(), 0)

            adv = (im + pert).clip(0, 255) 
            adv_list.append(adv)

        applied_patch.requires_grad = True
        mask_im_np = adv_list[-1].squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        mask_im_np = mask_im_np * patch_mask
        patch_loss_list = []
        patch_loss_list_np = []
        for model in ensemble:
            loss = model.loss(mask_im_np, applied_patch, bboxes_tgt, labels_tgt)
            patch_loss_list.append(loss)
            patch_loss_list_np.append(loss.item())
            LOSS[model.model_name].append(loss.item())
        
        patch_loss_ens = sum(w[i]*patch_loss_list[i] for i in range(len(ensemble)))
        patch_loss_ens.backward()

        with torch.no_grad():
            applied_patch = applied_patch - 5*torch.sign(applied_patch.grad)
            applied_patch = applied_patch.clamp(min=-eps, max=eps)
            applied_patch = applied_patch.masked_fill(torch.from_numpy(patch_mask.transpose(2, 0, 1)).unsqueeze(0).bool(), 0)

    applied_patch = applied_patch.squeeze().cpu().numpy()
    patch_tmp = np.zeros(patch.shape).astype(np.float64)
    for loc in applied_patch_loc:
        x, y = loc
        patch_tmp += applied_patch[:, y:y + patch.shape[2], x:x + patch.shape[1]]
    
    patch = patch_tmp / len(applied_patch_loc)
    patch = patch.astype(np.uint8)
    # TODO
    patch_adv_mask = applied_patch.transpose(1, 2, 0).astype(np.uint8) * np.logical_not(patch_mask)
    return adv_list, LOSS, patch, patch_mask, patch_adv_mask


def PM_tensor_weight_balancing_np(im_np, target, w_np, ensemble, eps, n_iters, alpha, dataset='voc', weight_balancing=False, adv_init=None, patch = None):
    """perturbation machine, numpy input version
    
    """
    device = next(ensemble[0].parameters()).device
    im = torch.from_numpy(im_np).permute(2,0,1).unsqueeze(0).float().to(device)
    if adv_init is None:
        adv = torch.clone(im) # adversarial image
    else:
        adv = torch.from_numpy(adv_init).permute(2,0,1).unsqueeze(0).float().to(device)

    # w = torch.from_numpy(w_np).float().to(device)
    adv_list, LOSS, patch, patch_mask, patch_adv_mask= PM_tensor_weight_balancing(im, adv, target, w_np, ensemble, eps, n_iters, alpha, dataset, weight_balancing, patch)
    adv_np = adv_list[-1].squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    # TODO
    adv_np = adv_np * patch_mask + patch_adv_mask
    return adv_np, LOSS, patch


def get_bb_loss(detections, target_clean, LOSS):
    """define the blackbox attack loss
        if the original object is detected, the loss is the conf score of the victim object
        otherwise, the original object disappears, the conf is below the threshold, the loss is the wb ensemble loss
    args:
        detections ():
        target_clean ():
        LOSS ():
    return:
        bb_loss (): the blackbox loss
    """
    max_iou = 0
    for items in detections:
        iou = get_iou(items, target_clean[0])
        if iou > max(max_iou, 0.3) and items[4] == target_clean[0][4]:
            max_iou = iou
            bb_loss = 1e3 + items[5] # add a large const to make sure it is larger than conf ens loss

    # if it disappears
    if max_iou < 0.3:
        bb_loss = LOSS['ens'][-1]

    return bb_loss


def save_det_to_fig(im_np, adv_np, LOSS, target_clean, all_models, im_id, im_idx, attack_goal, log_root, dataset, n_query):    
    """get the loss bb, success_list on all surrogate models, and save detections to fig
    
    args:

    returns:
        loss_bb (float): loss on the victim model
        success_list (list of 0/1s): successful for all models
    """
    fig_h = 5
    fig_w = 5
    n_all = len(all_models)
    fig, ax = plt.subplots(2,1+n_all,figsize=((1+n_all)*fig_w,2*fig_h))
    # 1st row, clean image, detection on surrogate models, detection on victim model
    # 2nd row, perturbed image, detection on surrogate models, detection on victim model
    row = 0
    ax[row,0].imshow(im_np)
    ax[row,0].set_title('clean image')
    for model_idx, model in enumerate(all_models):
        det_adv = model.det(im_np)

        indices_to_keep = np.any(det_adv[:, 4:5] == np.array(list(target_label_set)), axis=1)
        det_adv = det_adv[indices_to_keep]

        bboxes, labels, scores = det_adv[:,:4], det_adv[:,4], det_adv[:,5]
        vis_bbox(im_np, bboxes, labels, scores, ax=ax[row,model_idx+1], dataset=dataset)
        ax[row,model_idx+1].set_title(model.model_name)

    row = 1
    ax[row,0].imshow(adv_np)
    ax[row,0].set_title(f'adv image @ iter {n_query} \n {attack_goal}')
    success_list = [] # 1 for success, 0 for fail for all models
    for model_idx, model in enumerate(all_models):
        det_adv = model.det(adv_np)

        indices_to_keep = np.any(det_adv[:, 4:5] == np.array(list(target_label_set)), axis=1)
        det_adv = det_adv[indices_to_keep]

        bboxes, labels, scores = det_adv[:,:4], det_adv[:,4], det_adv[:,5]
        vis_bbox(adv_np, bboxes, labels, scores, ax=ax[row,model_idx+1], dataset=dataset)
        ax[row,model_idx+1].set_title(model.model_name)

        # check for success and get bb loss
        if model_idx == n_all-1:
            loss_bb = get_bb_loss(det_adv, target_clean, LOSS)

        # victim model is at the last index
        success_list.append(is_success(det_adv, target_clean))
    
    plt.tight_layout()
    if success_list[-1]:
        plt.savefig(log_root / f"{im_idx}_{im_id}_iter{n_query}_success.png")
    else:
        plt.savefig(log_root / f"{im_idx}_{im_id}_iter{n_query}.png")
    plt.close()

    return loss_bb, success_list
    
def patch_save_det_to_fig(im_np, adv_np, LOSS, target_clean, all_models, im_id, im_idx, attack_goal, log_root, dataset, n_query):    
    """get the loss bb, success_list on all surrogate models, and save detections to fig
    
    args:

    returns:
        loss_bb (float): loss on the victim model
        success_list (list of 0/1s): successful for all models
    """
    fig_h = 5
    fig_w = 5
    n_all = len(all_models)
    fig, ax = plt.subplots(2,1+n_all,figsize=((1+n_all)*fig_w,2*fig_h))
    # 1st row, clean image, detection on surrogate models, detection on victim model
    # 2nd row, perturbed image, detection on surrogate models, detection on victim model
    row = 0
    ax[row,0].imshow(im_np)
    ax[row,0].set_title('clean image')
    for model_idx, model in enumerate(all_models):
        det_adv = model.det(im_np)

        indices_to_keep = np.any(det_adv[:, 4:5] == np.array(list(target_label_set)), axis=1)
        det_adv = det_adv[indices_to_keep]

        bboxes, labels, scores = det_adv[:,:4], det_adv[:,4], det_adv[:,5]
        vis_bbox(im_np, bboxes, labels, scores, ax=ax[row,model_idx+1], dataset=dataset)
        ax[row,model_idx+1].set_title(model.model_name)

    row = 1
    ax[row,0].imshow(adv_np)
    ax[row,0].set_title(f'adv image @ iter {n_query} \n {attack_goal}')
    success_list = [] # 1 for success, 0 for fail for all models
    for model_idx, model in enumerate(all_models):
        det_adv = model.det(adv_np)

        indices_to_keep = np.any(det_adv[:, 4:5] == np.array(list(target_label_set)), axis=1)
        det_adv = det_adv[indices_to_keep]

        bboxes, labels, scores = det_adv[:,:4], det_adv[:,4], det_adv[:,5]
        vis_bbox(adv_np, bboxes, labels, scores, ax=ax[row,model_idx+1], dataset=dataset)
        ax[row,model_idx+1].set_title(model.model_name)

        # check for success and get bb loss
        if model_idx == n_all-1:
            loss_bb = get_bb_loss(det_adv, target_clean, LOSS)

        # victim model is at the last index
        success_list.append(is_success(det_adv, target_clean))
    
    plt.tight_layout()
    if success_list[-1]:
        plt.savefig(log_root / f"{im_idx}_{im_id}_iter{n_query}_success_patch.png")
    else:
        plt.savefig(log_root / f"{im_idx}_{im_id}_iter{n_query}_patch.png")
    plt.close()

    return loss_bb, success_list
  

def main():
    parser = argparse.ArgumentParser(description="generate perturbations")
    parser.add_argument("--eps", type=int, default=50, help="perturbation level: 10,20,30,40,50")
    parser.add_argument("--iters", type=int, default=20, help="number of inner iterations: 5,6,10,20...")
    # parser.add_argument("--gpu", type=int, default=0, help="GPU ID: 0,1")
    parser.add_argument("--root", type=str, default='result', help="the folder name of result")
    parser.add_argument("--victim", type=str, default='CO-DETR', help="victim model")
    parser.add_argument("--x", type=int, default=3, help="times alpha by x")
    parser.add_argument("--n_wb", type=int, default=6, help="number of models in the ensemble")
    parser.add_argument("--surrogate", type=str, default='YOLOv3', help="surrogate model when n_wb=1")
    # parser.add_argument("-untargeted", action='store_true', help="run untargeted attack")
    # parser.add_argument("--loss_name", type=str, default='cw', help="the name of the loss")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of w")
    parser.add_argument("--iterw", type=int, default=10, help="iterations of updating w")
    parser.add_argument("--dataset", type=str, default='coco', help="model dataset 'voc' or 'coco'. This will change the output range of detectors.")
    parser.add_argument("-single", action='store_true', help="only care about one obj")
    parser.add_argument("-no_balancing", action='store_true', help="do not balance weights at beginning")
    args = parser.parse_args()
    
    print(f"args.single: {args.single}")
    eps = args.eps
    n_iters = args.iters
    x_alpha = args.x
    alpha = eps / n_iters * x_alpha
    iterw = args.iterw
    n_wb = args.n_wb
    lr_w = args.lr
    dataset = args.dataset
    victim_name = args.victim

    # load surrogate models
    ensemble = []

    models_all = ['Faster R-CNN', 'YOLOv3', 'YOLOX', 'Sparse R-CNN','CO-DETR2', 'DETR']
    model_list = models_all[:n_wb]
    if n_wb == 1:
        model_list = [args.surrogate]
    for model_name in model_list:
        ensemble.append(model_train(model_name=model_name, dataset=dataset))

    # load victim model
    # ['RetinaNet', 'Libra', 'FoveaBox', 'FreeAnchor', 'DETR', 'Deformable']
    if victim_name == 'Libra':
        victim_name = 'Libra R-CNN'
    elif victim_name == 'Deformable':
        victim_name = 'Deformable DETR'

    model_victim = model_train(model_name=victim_name, dataset=dataset)

    all_models = ensemble + [model_victim]

    # create folders
    exp_name = f'BB_{n_wb}wb_linf_{eps}_iters{n_iters}_alphax{x_alpha}_victim_{victim_name}_lr{lr_w}_iterw{iterw}'
    if dataset != 'voc':
        exp_name += f'_{dataset}'
    if n_wb == 1:
        exp_name += f'_{args.surrogate}'
    if args.single:
        exp_name += '_single'
    if args.no_balancing:
        exp_name += '_noBalancing'

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M")
    exp_name += f'_{formatted_time}'
    print(f"\nExperiment: {exp_name} \n")
    result_root = Path(f"results_detection_voc/phase2_result/")
    exp_root = result_root / exp_name
    log_root = exp_root / 'logs'
    log_root.mkdir(parents=True, exist_ok=True)
    log_loss_root = exp_root / 'logs_loss'
    log_loss_root.mkdir(parents=True, exist_ok=True)
    adv_root = exp_root / 'advs'
    adv_root.mkdir(parents=True, exist_ok=True)
    target_root = exp_root / 'targets'
    target_root.mkdir(parents=True, exist_ok=True)

    if dataset == 'voc':
        label_names = VOC_BBOX_LABEL_NAMES
    else:
        label_names = COCO_BBOX_LABEL_NAMES

    dict_k_sucess_id_v_query = {} # query counts of successful im_ids
    dict_k_valid_id_v_success_list = {} # lists of success for all mdoels for valid im_ids
    n_obj_list = []

    test_image_ids = JSON.load(open(f"data/phase2.json"))
    # patch = patch_initialization((3, 1912, 1028))
    patch = np.load("patch/patch3.npy")
    for iter in range(3):
        for im_idx, im_id in tqdm(enumerate(test_image_ids[:100])):
            im_root = Path("data/test_phase2")
            im_path = im_root / f"{im_id}.jpg"
            im_np = np.array(Image.open(im_path).convert('RGB'))
            
            # get detection on clean images and determine target class
            det = model_victim.det(im_np)
            # only attack specify objects, car, person, traffic light and etc
            indices_to_keep = np.any(det[:, 4:5] == np.array(list(target_label_set)), axis=1)
            det = det[indices_to_keep]

            bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
            print(f"n_objects: {len(det)}")
            n_obj_list.append(len(det))
            if len(det) == 0: # if nothing is detected, skip this image
                adv_path = adv_root / f"{im_id}.jpg"
                adv_png = Image.fromarray(im_np.astype(np.uint8))
                adv_png.save(adv_path)
                continue
            else:
                dict_k_valid_id_v_success_list[im_id] = []

            target = det.copy()
            attack_goal = "attack all objects to potted plant"
            for victim_idx in range(0, len(det)):

                victim_class = int(det[victim_idx,4])

                # attack object to potted plant
                target_class = 58  # potted plant

                target[victim_idx, 4] = target_class

            info = f"im_idx: {im_idx}, im_id: {im_id}, victim_class: {label_names[victim_class]}, target_class: {label_names[target_class]}\n"
            print(info)
            file = open(exp_root / f'{exp_name}.txt', 'a')
            file.write(f"{info}\n\n")
            file.close()

            # only keep one label
            target_clean = target[victim_idx,:][None]

            if args.single: # only care about the target object
                target = target_clean
            # save target to np
            np.save(target_root/f"{im_id}_target", target)     
            

            if args.no_balancing:
                print(f"no_balancing, using equal weights")
                w_inv = np.ones(n_wb) 
                w_np = np.ones(n_wb) / n_wb
            else:
                # determine the initial w, via weight balancing
                dummy_w = np.ones(n_wb)
                _, LOSS, patch = PM_tensor_weight_balancing_np(im_np, target, dummy_w, ensemble, eps, n_iters=1, alpha=alpha, dataset=dataset, patch=patch)
                loss_list_np = [LOSS[name][0] for name in model_list]
                w_inv = 1 / np.array(loss_list_np)
                w_np = w_inv / w_inv.sum()
                print(f"w_np: {w_np}")
            np.save(target_root/f"{im_id}_patch", patch)

            adv_np, LOSS, patch = PM_tensor_weight_balancing_np(im_np, target, w_np, ensemble, eps, n_iters, alpha=alpha, dataset=dataset, patch=patch)
            np.save(target_root/f"{im_id}_patch", patch)
            n_query = 0
            loss_bb, success_list = save_det_to_fig(im_np, adv_np, LOSS, target_clean, all_models, im_id, im_idx, attack_goal, log_root, dataset, n_query)
            dict_k_valid_id_v_success_list[im_id].append(success_list)

            # save adv in folder
            adv_path = adv_root / f"{im_id}.jpg"
            adv_png = Image.fromarray(adv_np.astype(np.uint8))
            adv_png.save(adv_path)

            # stop whenever successful
            if success_list[-1]:
                dict_k_sucess_id_v_query[im_id] = n_query
                print(f"success! image im idx: {im_idx}")
                
                w_list = []
                loss_bb_list = [loss_bb]
                loss_ens_list = LOSS['ens'] # ensemble losses during training
            else:

                n_query += 1

                w_list = []        
                loss_bb_list = [loss_bb]
                loss_ens_list = LOSS['ens'] # ensemble losses during training

                idx_w = 0 # idx of wb in W, rotate
                while n_query < iterw:

                    ##################################### query plus #####################################
                    w_np_temp_plus = w_np.copy()
                    w_np_temp_plus[idx_w] += lr_w * w_inv[idx_w]
                    adv_np_plus, LOSS_plus, patch = PM_tensor_weight_balancing_np(im_np, target, w_np_temp_plus, ensemble, eps, n_iters, alpha=alpha, dataset=dataset, adv_init=adv_np, patch=patch)
                    np.save(target_root/f"{im_id}_patch", patch)
                    loss_bb_plus, success_list = save_det_to_fig(im_np, adv_np_plus, LOSS_plus, target_clean, all_models, im_id, im_idx, attack_goal, log_root, dataset, n_query)

                    dict_k_valid_id_v_success_list[im_id].append(success_list)

                    n_query += 1
                    print(f"iter: {n_query}, {idx_w} +, loss_bb: {loss_bb_plus}")

                    # save adv in folder
                    adv_path = adv_root / f"{im_id}.jpg"
                    adv_png = Image.fromarray(adv_np_plus.astype(np.uint8))
                    adv_png.save(adv_path)

                    # stop whenever successful
                    if success_list[-1]:
                        dict_k_sucess_id_v_query[im_id] = n_query
                        print(f"success! image im idx: {im_idx}")
                        loss_bb = loss_bb_plus
                        loss_ens = LOSS_plus["ens"]
                        w_np = w_np_temp_plus
                        adv_np = adv_np_plus
                        break

                    #######################################################################################
                    

                    ##################################### query minus #####################################
                    w_np_temp_minus = w_np.copy()
                    w_np_temp_minus[idx_w] -= lr_w * w_inv[idx_w]
                    adv_np_minus, LOSS_minus, patch = PM_tensor_weight_balancing_np(im_np, target, w_np_temp_minus, ensemble, eps, n_iters, alpha=alpha, dataset=dataset, adv_init=adv_np, patch=patch)
                    np.save(target_root/f"{im_id}_patch", patch)
                    loss_bb_minus, success_list = save_det_to_fig(im_np, adv_np_minus, LOSS_minus, target_clean, all_models, im_id, im_idx, attack_goal, log_root, dataset, n_query)

                    dict_k_valid_id_v_success_list[im_id].append(success_list)

                    n_query += 1
                    print(f"iter: {n_query}, {idx_w} -, loss_bb: {loss_bb_minus}")

                    # save adv in folder
                    adv_path = adv_root / f"{im_id}.jpg"
                    adv_png = Image.fromarray(adv_np_minus.astype(np.uint8))
                    adv_png.save(adv_path)

                    # stop whenever successful
                    if success_list[-1]:
                        dict_k_sucess_id_v_query[im_id] = n_query
                        print(f"success! image im idx: {im_idx}")
                        loss_bb = loss_bb_minus
                        loss_ens = LOSS_minus["ens"]
                        w_np = w_np_temp_minus
                        adv_np = adv_np_minus
                        break

                    #######################################################################################


                    ##################################### update w, adv #####################################
                    if loss_bb_plus < loss_bb_minus:
                        loss_bb = loss_bb_plus
                        loss_ens = LOSS_plus["ens"]
                        w_np = w_np_temp_plus
                        adv_np = adv_np_plus
                    else:
                        loss_bb = loss_bb_minus
                        loss_ens = LOSS_minus["ens"]
                        w_np = w_np_temp_minus
                        adv_np = adv_np_minus

                    # relu and normalize
                    w_np = np.maximum(0, w_np)
                    w_np = w_np + 0.005 # minimum set to 0.005
                    w_np = w_np / w_np.sum()
                    #######################################################################################
                        
                    idx_w = (idx_w+1)%n_wb
                    w_list.append(w_np.tolist())
                    loss_bb_list.append(loss_bb)
                    loss_ens_list += loss_ens


if __name__ == '__main__':
    main()
