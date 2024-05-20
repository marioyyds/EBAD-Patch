# detection models pre-trained on coco dataset
# https://mmdetection.readthedocs.io/en/stable/model_zoo.html

model_info = {
    'Faster R-CNN': {
        'config_file': 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    },
    'YOLOv3': {
        'config_file': 'configs/yolo/yolov3_d53_8xb8-ms-416-273e_coco.py',
        'checkpoint_file': 'checkpoints/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'
    },
    'DETR': {
        'config_file': 'configs/detr/detr_r50_8xb2-150e_coco.py',
        'checkpoint_file': 'checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
    },
    'Deformable DETR': {
        'config_file': 'configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py',
        'checkpoint_file': 'checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
    },
    'Grid R-CNN': {
        'config_file': 'configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py',
        'checkpoint_file': 'checkpoints/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth'
    },
    'YOLOX': {
        'config_file': 'configs/yolox/yolox_tiny_8xb8-300e_coco.py',
        'checkpoint_file': 'checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
    },
    'CO-DETR': {
        'config_file': 'projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py',
        'checkpoint_file': 'checkpoints/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'
    },
    'Sparse R-CNN': {
        'config_file': 'configs/sparse_rcnn/sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco.py',
        'checkpoint_file': 'checkpoints/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth'
    }
}


def main():
    import urllib.request
    from pathlib import Path

    checkpoints_root = Path('mmdetection/checkpoints')
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    existing_files = list(checkpoints_root.glob('*.pth'))
    existing_files = [file.name for file in existing_files]

    for idx,model_name in enumerate(model_info):
        url = model_info[model_name]['download_link']
        file_name = url.split('/')[-1]
        if file_name in existing_files:
            print(f"{model_name} already exists, {idx+1}/{len(model_info)}")
            continue
        print(f'downloading {model_name} {idx+1}/{len(model_info)}')
        file_data = urllib.request.urlopen(url).read()
        with open(checkpoints_root / file_name, 'wb') as f:
            f.write(file_data)


if __name__ == "__main__":
    main()