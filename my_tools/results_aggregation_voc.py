import pandas as pd
import glob
import os
import json

CORRUPTIONS = ['defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression','gaussian_noise', 'shot_noise', 'impulse_noise']
CO_LEVEL = ['1','2','3','4','5']
pattern = 'pascal_voc/mAP'

data_root = 'local_results/od/coco/fasterRCNN_r50_fpn_1x_original'

imc_index = ['clean']+CORRUPTIONS
imc_columns = CO_LEVEL
imc_df = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)

for co in imc_index:
    
    for l in imc_columns:
        
        if co == 'clean':
            data_path = glob.glob(os.path.join(data_root,co,'*/*.json'))[0]
            
            
            contents = []
            with open(data_path, 'r') as file:  
                for line in file.readlines():
                    dic = json.loads(line)
                    contents.append(dic)
            
            value = contents[-1][pattern]
            imc_df.loc[co] = round(value*100,1)
            break
        else:
            data_path = glob.glob(os.path.join(data_root,co,l,'*/*.json'))[0]
            
            contents = []
            with open(data_path, 'r') as file:  
                for line in file.readlines():
                    dic = json.loads(line)
                    contents.append(dic)
            
            value = contents[-1][pattern]
            imc_df.loc[co,l] = round(value*100,1)

imc_df['avg'] = imc_df.mean(axis=1).round(1)
average_excluding_first_row = imc_df[1:].avg.mean()

total_index = ['VOC']
total_columns = ['VOC/AP50']
total_df = pd.DataFrame(data=0, columns=total_columns, index=total_index)

total_df.loc['VOC','VOC/AP50'] = round(imc_df.loc['clean','1'],1)
total_df.loc['VOC-C','VOC/AP50'] = round(average_excluding_first_row,1)
imc_df_path = os.path.join(data_root,'detailed.csv')
total_path = os.path.join(data_root,'total.csv')

imc_df.to_csv(imc_df_path)
total_df.to_csv(total_path)
print('Done!')