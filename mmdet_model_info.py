# detection models pre-trained on coco dataset
# https://mmdetection.readthedocs.io/en/stable/model_zoo.html

model_info = {
    'Faster R-CNN': {
        'config_file': 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    },
    'RetinaNet': {
        'config_file': 'configs/retinanet/retinanet_r50_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
    },
    'SSD': {
        'config_file': 'configs/ssd/ssd512_coco.py',
        'checkpoint_file': 'checkpoints/ssd512_coco_20210803_022849-0a47a1ca.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth'
        
    },
    'YOLOv3': {
        'config_file': 'configs/yolo/yolov3_d53_8xb8-ms-416-273e_coco.py',
        'checkpoint_file': 'checkpoints/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'
    },
    'GN+WS': {
        'config_file': 'configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py',
        'checkpoint_file': 'checkpoints/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth'
    },
    'Libra R-CNN': {
        'config_file': 'configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth'
    },
    'FCOS': {
        'config_file': 'configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco.py',
        'checkpoint_file': 'checkpoints/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth'
    },
    'FoveaBox': {
        'config_file': 'configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py',
        'checkpoint_file': 'checkpoints/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth'
    },
    'FreeAnchor': {
        'config_file': 'configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth'
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
    'Guided Anchoring': {
        'config_file': 'configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/ga_faster_r50_caffe_fpn_1x_coco_20200702_000718-a11ccfe6.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco/ga_faster_r50_caffe_fpn_1x_coco_20200702_000718-a11ccfe6.pth'
    },
    'RepPoints': {
        'config_file': 'configs/reppoints/reppoints_moment_r50_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_1x_coco/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth'
    },
    'Grid R-CNN': {
        'config_file': 'configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py',
        'checkpoint_file': 'checkpoints/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth'
    },
    'GHM': {
        'config_file': 'configs/ghm/retinanet_ghm_r50_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r50_fpn_1x_coco/retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth'
    },
    'GCNet': {
        'config_file': 'configs/gcnet/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco.py',
        'checkpoint_file': 'checkpoints/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco_20200515_211915-187da160.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco_20200515_211915-187da160.pth'
    },
    'CenterNet': {
        'config_file': 'configs/centernet/centernet_resnet18_dcnv2_140e_coco.py',
        'checkpoint_file': 'checkpoints/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth'
    },
    'YOLOX': {
        'config_file': 'configs/yolox/yolox_tiny_8xb8-300e_coco.py',
        'checkpoint_file': 'checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
    },
    'YOLOF': {
        'config_file': 'configs/yolof/yolof_r50_c5_8x8_1x_coco.py',
        'checkpoint_file': 'checkpoints/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth'
    },
    'ATSS': {
        'config_file': 'configs/atss/atss_r50_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'
    },
    'CO-DETR': {
        'config_file': 'projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py',
        'checkpoint_file': 'checkpoints/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'
    },
    'ViTDET': {
        'config_file': 'projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py',
        'checkpoint_file': 'checkpoints/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v3.0/vitdet/vitdet_mask-rcnn_vit-b-mae_lsj-100e/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth'
    },
    'GLIP': {
        'config_file': 'configs/glip/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco.py',
        'checkpoint_file': 'checkpoints/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_100800-e9be4274.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_100800-e9be4274.pth'
    },
    'CO-DETR1': {
        'config_file': 'projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_1x_coco.py',
        'checkpoint_file': 'checkpoints/co_dino_5scale_swin_large_1x_coco-27c13da4.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_1x_coco-27c13da4.pth'
    },
    'CO-DETR2': {
        'config_file': 'projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_3x_coco.py',
        'checkpoint_file': 'checkpoints/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth'
    },
    'TOOD': {
        'config_file': 'configs/tood/tood_r50_fpn_ms-2x_coco.py',
        'checkpoint_file': 'checkpoints/tood_r50_fpn_mstrain_2x_coco_20211210_144231-3b23174c.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r50_fpn_mstrain_2x_coco/tood_r50_fpn_mstrain_2x_coco_20211210_144231-3b23174c.pth'
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