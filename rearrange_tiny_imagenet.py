import os
import glob

"""
Move image files to agree with directory of form : /root/[class]/[img_id].jpeg
This is required to use torchvision.ImageFolder for convenience.

Currently, tiny imagenet has directory of form : /root/[class]/images/[img_id].jpeg
We need to move image files to its parent directory.
"""

IMAGENET_DIR = 'tiny-imagenet-200'

for root, dirs, files in os.walk(IMAGENET_DIR):
    if 'train' in root and 'images' in root:
        class_dir, _ = os.path.split(root)
        print('moving for : {}'.format(class_dir))

        # remove annotation files
        for txtfile in glob.glob(os.path.join(class_dir, '*.txt')):
            os.remove(txtfile)

        # move image files to parent directory
        for img_file in files:
            original_path = os.path.join(root, img_file)
            new_path = os.path.join(class_dir, img_file)
            os.rename(original_path, new_path)
        os.rmdir(root)
