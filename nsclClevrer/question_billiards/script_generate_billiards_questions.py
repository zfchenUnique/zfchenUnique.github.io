import cv2
from modules.video_parser import Simulation 
from modules.opts import options 
import numpy as np
import os
import pickle
import pdb
from scipy.optimize import linear_sum_assignment
from utils.util import get_sub_file_list, set_debugger, pickledump, visualize_specific_frame  

set_debugger()

def parse_color_info(opts):
    ann_full_list = get_sub_file_list(opts.ann_dir, file_type='pkl')
    ann_full_list = sorted(ann_full_list)
    for idx, ann_path in enumerate(ann_full_list):
        with open(ann_path, 'rb') as f:
            ann = pickle.load(f)
        vid_str = os.path.basename(ann_path).split('.')[0]
        frm_num = ann.shape[0]
        obj_num = ann.shape[1]
        obj_rgb_list = [np.array([0, 0, 0], dtype=np.float64) for idx in range(obj_num)]
        #red_value = np.array((255,0,0))
        #yellow_value = np.array((255,255,0))
        #white_value = np.array((255, 255, 255))
       
        color_list = ['red', 'yellow', 'white']
        red_value = np.array((73,21,61))
        yellow_value = np.array((105,80,80))
        white_value = np.array((100, 89, 102))

        target_color_map = [red_value, yellow_value, white_value]
        obj_buffer_list =  [[] for idx in range(obj_num)]
        for frm_id in range(frm_num):
            full_rgb_path = os.path.join(opts.ann_dir, vid_str, 
                    str(frm_id).zfill(3)+'.jpg')
            img = cv2.imread(full_rgb_path)
            b, g, r = cv2.split(img)   
            img2 = cv2.merge([r, g, b])
            for obj_id in range(obj_num):
                tmp_box = ann[frm_id, obj_id]
                x1, y1, x2, y2 = int(tmp_box[1]), int(tmp_box[2]), int(tmp_box[3])+1, int(tmp_box[4])+1    
                tmp_patch = img2[y1:y2, x1:x2]
                mean_obj_rgb = tmp_patch.mean(axis=0).mean(axis=0)
                obj_rgb_list[obj_id] +=mean_obj_rgb
                obj_buffer_list[obj_id].append(mean_obj_rgb)
        for obj_id, obj_rgb in enumerate(obj_rgb_list):
            obj_rgb_list[obj_id] = obj_rgb / frm_num 
        cost_matrix = np.zeros((obj_num, obj_num)) 
        for obj_id, obj_val in enumerate(obj_rgb_list):
            for color_id, color_val in enumerate(target_color_map):
                tmp_diff = obj_val - color_val
                cost_matrix[obj_id, color_id] = np.sqrt(np.sum(tmp_diff**2))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        tmp_color_list = ['', '', '']
        for obj_idx in range(obj_num):
            obj_id = row_ind[obj_idx]
            color_id = col_ind[obj_idx]
            tmp_color_list[obj_id] = color_list[color_id]
        if not os.path.isdir(opts.attr_dir):
            os.makedirs(opts.attr_dir)
        out_path = os.path.join(opts.attr_dir, vid_str+'.pkl') 
        pickledump(out_path, tmp_color_list)
        
        if not os.path.isdir(opts.visualize_dir):
            os.makedirs(opts.visualize_dir)
        out_img_path = os.path.join(opts.visualize_dir, vid_str+'.png') 
        color_dict = {color:target_color_map[c_id] for c_id, color in enumerate(color_list)} 
        visualize_specific_frame(img, ann[-1,  :, 1:], tmp_color_list, out_img_path, color_dict)
        #pdb.set_trace()


if __name__=='__main__':
    option = options()
    opts = option.parse()
    if opt.parse_color_flag ==1:
        parse_color_info(opts)
    if opt.generate_ques_flag ==1:
        sim = Simulation(opts.ann_dir, 0) 
        pdb.set_trace()
