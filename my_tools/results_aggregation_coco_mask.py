import pandas as pd
import glob
import os
import json

CORRUPTIONS = ['defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression','gaussian_noise', 'shot_noise', 'impulse_noise']
CO_LEVEL = ['1','2','3','4','5']
pattern = 'coco/bbox_mAP'
pattern_s = 'coco/bbox_mAP_s'
pattern_m = 'coco/bbox_mAP_m'
pattern_l = 'coco/bbox_mAP_l'

pattern_seg = 'coco/segm_mAP'
pattern_seg_s = 'coco/segm_mAP_s'
pattern_seg_m = 'coco/segm_mAP_m'
pattern_seg_l = 'coco/segm_mAP_l'

data_root = 'local_results/od/coco/cascade-mask-rcnn_r50_fpn_1x_coco_suppressor'

imc_index = ['clean']+CORRUPTIONS
imc_columns = CO_LEVEL
imc_df = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)

imc_df_s = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)
imc_df_m = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)
imc_df_l = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)

imc_df_seg = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)

imc_df_seg_s = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)
imc_df_seg_m = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)
imc_df_seg_l = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)

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

            value_s = contents[-1][pattern_s]
            value_m = contents[-1][pattern_m]
            value_l = contents[-1][pattern_l]
            
            value_seg = contents[-1][pattern_seg]

            value_seg_s = contents[-1][pattern_seg_s]
            value_seg_m = contents[-1][pattern_seg_m]
            value_seg_l = contents[-1][pattern_seg_l]

            imc_df_s.loc[co] = round(value_s*100,1)
            imc_df_m.loc[co] = round(value_m*100,1)
            imc_df_l.loc[co] = round(value_l*100,1)
            
            imc_df.loc[co] = round(value*100,1)

            imc_df_seg.loc[co] = round(value_seg*100,1)
            
            imc_df_seg_s.loc[co] = round(value_seg_s*100,1)
            imc_df_seg_m.loc[co] = round(value_seg_m*100,1)
            imc_df_seg_l.loc[co] = round(value_seg_l*100,1)

            

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

            value_s = contents[-1][pattern_s]
            value_m = contents[-1][pattern_m]
            value_l = contents[-1][pattern_l]

            imc_df_s.loc[co,l] = round(value_s*100,1)
            imc_df_m.loc[co,l] = round(value_m*100,1)
            imc_df_l.loc[co,l] = round(value_l*100,1)
            
            value_seg = contents[-1][pattern_seg]

            value_seg_s = contents[-1][pattern_seg_s]
            value_seg_m = contents[-1][pattern_seg_m]
            value_seg_l = contents[-1][pattern_seg_l]
            
            imc_df_seg.loc[co,l] = round(value_seg*100,1)
            
            imc_df_seg_s.loc[co,l] = round(value_seg_s*100,1)
            imc_df_seg_m.loc[co,l] = round(value_seg_m*100,1)
            imc_df_seg_l.loc[co,l] = round(value_seg_l*100,1)

imc_df['avg'] = imc_df.mean(axis=1).round(1)
average_excluding_first_row = imc_df[1:].avg.mean()

imc_df_s['avg'] = imc_df_s.mean(axis=1).round(1)
average_excluding_first_row_s = imc_df_s[1:].avg.mean()

imc_df_m['avg'] = imc_df_m.mean(axis=1).round(1)
average_excluding_first_row_m = imc_df_m[1:].avg.mean()

imc_df_l['avg'] = imc_df_l.mean(axis=1).round(1)
average_excluding_first_row_l = imc_df_l[1:].avg.mean()

imc_df_seg['avg'] = imc_df_seg.mean(axis=1).round(1)
average_excluding_first_row_seg = imc_df_seg[1:].avg.mean()

imc_df_seg_s['avg'] = imc_df_seg_s.mean(axis=1).round(1)
average_excluding_first_row_seg_s = imc_df_seg_s[1:].avg.mean()

imc_df_seg_m['avg'] = imc_df_seg_m.mean(axis=1).round(1)
average_excluding_first_row_seg_m = imc_df_seg_m[1:].avg.mean()

imc_df_seg_l['avg'] = imc_df_seg_l.mean(axis=1).round(1)
average_excluding_first_row_seg_l = imc_df_seg_l[1:].avg.mean()

total_index = ['COCO']
total_columns = ['coco/bbox_mAP']
total_df = pd.DataFrame(data=0, columns=total_columns, index=total_index)

total_df.loc['COCO','coco/bbox_mAP'] = round(imc_df.loc['clean','1'],1)
total_df.loc['COCO-C','coco/bbox_mAP'] = round(average_excluding_first_row,1)
total_df.loc['COCO-C_s','coco/bbox_mAP'] = round(average_excluding_first_row_s,1)
total_df.loc['COCO-C_m','coco/bbox_mAP'] = round(average_excluding_first_row_m,1)
total_df.loc['COCO-C_l','coco/bbox_mAP'] = round(average_excluding_first_row_l,1)

total_index = ['COCO']
total_columns = ['coco/segm_mAP']
total_df_seg = pd.DataFrame(data=0, columns=total_columns, index=total_index)

total_df_seg.loc['COCO','coco/segm_mAP'] = round(imc_df_seg.loc['clean','1'],1)
total_df_seg.loc['COCO-C','coco/segm_mAP'] = round(average_excluding_first_row_seg,1)
total_df_seg.loc['COCO-C_s','coco/segm_mAP'] = round(average_excluding_first_row_seg_s,1)
total_df_seg.loc['COCO-C_m','coco/segm_mAP'] = round(average_excluding_first_row_seg_m,1)
total_df_seg.loc['COCO-C_l','coco/segm_mAP'] = round(average_excluding_first_row_seg_l,1)

imc_df_path = os.path.join(data_root,'detailed_det.csv')
imc_df_path_s = os.path.join(data_root,'detailed_det_s.csv')
imc_df_path_m = os.path.join(data_root,'detailed_det_m.csv')
imc_df_path_l = os.path.join(data_root,'detailed_det_l.csv')

total_path = os.path.join(data_root,'total_det.csv')

imc_df_path_seg = os.path.join(data_root,'detailed_seg.csv')
imc_df_path_seg_s = os.path.join(data_root,'detailed_seg_s.csv')
imc_df_path_seg_m = os.path.join(data_root,'detailed_seg_m.csv')
imc_df_path_seg_l = os.path.join(data_root,'detailed_seg_l.csv')

total_path_seg = os.path.join(data_root,'total_seg.csv')

imc_df.to_csv(imc_df_path)
imc_df_s.to_csv(imc_df_path_s)
imc_df_m.to_csv(imc_df_path_m)
imc_df_l.to_csv(imc_df_path_l)

imc_df_seg.to_csv(imc_df_path_seg)
imc_df_seg_s.to_csv(imc_df_path_seg_s)
imc_df_seg_m.to_csv(imc_df_path_seg_m)
imc_df_seg_l.to_csv(imc_df_path_seg_l)


total_df.to_csv(total_path)
total_df_seg.to_csv(total_path_seg)
print('Done!')