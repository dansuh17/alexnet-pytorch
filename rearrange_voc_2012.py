import os
import os.path as path


VOC_ROOT = path.join('./', 'VOCdevkit', 'VOC2012')
VOC_LABEL_DIR = path.join(VOC_ROOT, 'ImageSets', 'Main')
VOC_IMG_DIR = path.join(VOC_ROOT, 'JPEGImages')
VOC_OUT_REARRANGED = path.join('./', 'voc-data-rearr')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def move_img(img_name, cls_name):
    img_path = path.join(VOC_IMG_DIR, img_name + '.jpg')
    out_img_path = path.join(VOC_OUT_REARRANGED, cls_name)
    os.system('cp {} {}'.format(img_path, out_img_path))


# create output directory
if not os.path.exists(VOC_OUT_REARRANGED):
    os.mkdir(VOC_OUT_REARRANGED)

for clsname in CLASSES:
    class_folder = path.join(VOC_OUT_REARRANGED, clsname)
    if not os.path.exists(class_folder):
        os.mkdir(class_folder)

for filename in os.listdir(VOC_LABEL_DIR):
    if filename in ['val.txt', 'train.txt', 'trainval.txt']:
        continue

    cls_traintype, _ = path.splitext(filename)
    clsname, traintype = cls_traintype.split('_')

    if traintype != 'train':
        continue
  
    print('Processing: {}'.format(filename))
    with open(path.join(VOC_LABEL_DIR, filename), 'r') as f:
        lines = f.readlines()
        for line in lines:
          img_name, is_cls = line.strip().split()
          if is_cls == '1':
              move_img(img_name, clsname)
