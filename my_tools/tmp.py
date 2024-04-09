import os
import shutil
import glob

data_path = 'local_data/datasets/object_detection/COCO2017/train2017'


data_list = glob.glob(os.path.join(data_path,'*.png'))

for i in data_list:

    os.remove(i)

print('Done!')