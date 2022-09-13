# pilot.phil aka dwb
# 2022-09-06
# copy command to console to run

# 1.evaluate
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --images=input:output

python eval.py --trained_model=weights/yolact_resnet50_ice_channel_config_2666_40000.pth --score_threshold=0.15 --top_k=15 --images=input:output

python eval.py --trained_model=weights/yolact_resnet50_zhongliu_config_222_3108_interrupt.pth --score_threshold=0.15 --top_k=15 --images=input:output

python eval_images_yolact.py --trained_model=weights/yolact_resnet50_zhongliu_13333_160000.pth \
--score_threshold=0.15 \
--top_k=15 \
--images=/home/dwb/Pictures/zl_test/in:/home/dwb/Pictures/zl_test/out \
--config=yolact_resnet50_zhongliu \
--json_path='/home/dwb/Pictures/zl_test/out'


# 2.train
# reference: https://www.immersivelimit.com/tutorials/train-yolact-with-a-custom-coco-dataset
# 2.1 convert labelme format to coco format
python ./labelme2coco.py --input_dir ./dataset/channel --output_dir ./dataset/channel_opt --labels ./dataset/labels.txt
# labels.txt must be like,start with specified head
 __ignore__
 __background__
 person
 bicycle
 car
 motorbike

# 2.2 modify config file
# this file is ./data/config.py

# in DATASETS section
# add you dataset trainning function
# like that:
 cig_butts_dataset = dataset_base.copy({
   'name': 'Immersive Limit - Cigarette Butts',
   'train_info': '<path to dataset>/cig_butts/train/coco_annotations.json',
   'train_images': '<path to dataset>/cig_butts/train/images/',
   'valid_info': '<path to dataset>/cig_butts/val/coco_annotations.json',
   'valid_images': '<path to dataset>/cig_butts/val/images/',
   'class_names': ('cig_butt'),
   'label_map': { 1:  1 }
 })

# in YOLACT v1.0 CONFIGS section
# add config function
# like that
 yolact_resnet50_cig_butts_config = yolact_resnet50_config.copy({
     'name': 'yolact_plus_resnet50_cig_butts',
     # Dataset stuff
     'dataset': cig_butts_dataset,
     'num_classes': len(cig_butts_dataset.class_names) + 1,

     # Image Size
     'max_size': 512,
 })

# 2.3 train
python ./train.py --config=yolact_resnet50_cig_butts_config






#zhongliu_dataset = dataset_base.copy({
#    'name': 'zhongliu',
#
#    'train_images': '/media/dwb/media/cloned/yolact/dataset/ZL_opt',
#    'train_info':   '/media/dwb/media/cloned/yolact/dataset/ZL_opt/annotations.json',
#
#    'valid_images': '/media/dwb/media/cloned/yolact/dataset/ZL_opt',
#    'valid_info':   '/media/dwb/media/cloned/yolact/dataset/ZL_opt/annotations.json',
#
#    'has_gt': True,
#    'class_names': ('channel')
#})


