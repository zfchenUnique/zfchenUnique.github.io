import cv2
import json
import numpy as np
import pdb
import pickle
import os
from scipy.optimize import linear_sum_assignment

def set_debugger():
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

def parse_video_info_v2(opts, vid):
    img_fn = os.path.join(opts.ann_dir, 'img_concat_'+str(vid)+'.png')
    #mask_color_fn = os.path.join(opts.ann_dir, 'mask_order_'+str(vid)+'.png')
    mask_color_fn = os.path.join(opts.ann_dir, 'mask_color_'+str(vid)+'.png')
    mask_color_concat = cv2.imread(mask_color_fn)
    H_concat, W, C = mask_color_concat.shape
    img_concat = cv2.imread(img_fn)

    H = 224
    assert H_concat % H ==0
    assert H==W
    img_num = H_concat // H
    kernel = np.ones((5,5), np.uint8)
    video_attr = {}

    # BGR
    red_value = np.array((0,0, 255))
    yellow_value = np.array((0,255,255))
    green_value = np.array((0, 255, 0))
    blue_value = np.array((255, 0, 0))
    color_list = ['red', 'yellow', 'green', 'blue']
    target_color_map = [red_value, yellow_value, green_value, blue_value ]
    color_dict = {color:target_color_map[c_id] for c_id, color in enumerate(color_list)} 
        
    mask_color = mask_color_concat[:H] 
    obj_num = np.max(mask_color)
    obj_buffer_list =  [[] for idx in range(obj_num)]
    obj_rgb_list = [np.array([0, 0, 0], dtype=np.float64) for idx in range(obj_num)]

    img_box_list = []
    for img_id in range(img_num):
        st_row_id = img_id * H
        ed_row_id = img_id * H + H
        mask_color = mask_color_concat[st_row_id: ed_row_id] 
        img = img_concat[st_row_id: ed_row_id] 
        max_id = np.max(mask_color)
        tmp_box_list = []
        for obj_id in range(1, max_id+1):
            tmp_mask  = ((mask_color==obj_id)*255).astype(np.uint8)
            tmp_mask = cv2.morphologyEx(tmp_mask, cv2.MORPH_OPEN, kernel)
            mask = np.sum(tmp_mask, axis=2)  
            if np.sum(mask)==0:
                tmp_box = np.array([0, 0, 0, 0])
                tmp_box_list.append(tmp_box)
                continue 
            ans = cv2.connectedComponentsWithStats(tmp_mask[:, :, 0])
            #pdb.set_trace()
            rgn_num = ans[2].shape[0]
            sort_idx = np.argsort(-ans[2][:, -1])
            for idx in range(rgn_num):
                idx2 = sort_idx[idx]
                if np.array_equal(ans[2][idx2, :4], np.array([0, 0, 224, 224])):
                    continue
                else:
                    break 
            tmp_box_xywh = ans[2][idx2, :4]
            x1 = tmp_box_xywh[0]
            y1 = tmp_box_xywh[1]
            x2 = x1  + tmp_box_xywh[2]
            y2 = y1  + tmp_box_xywh[3]
            tmp_box = np.array([x1, y1, x2, y2])
            tmp_box_list.append(tmp_box)
            bi_mask  = tmp_mask == 255
            pos_pixel_num = np.sum(bi_mask[:, :, 0])
            mean_obj_rgb = 1/(0.000001+pos_pixel_num) * np.sum((img * bi_mask).reshape(-1, C), axis=0)
            obj_rgb_list[obj_id-1] +=mean_obj_rgb
            obj_buffer_list[obj_id-1].append(mean_obj_rgb)
            
            if opts.debug:
                tmp_mask = cv2.rectangle(tmp_mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.imwrite('dumps/visualization/mask/'+str(img_id)+'_'+str(obj_id-1)+'.png', tmp_mask.astype(np.uint8))
            #pdb.set_trace()
        tmp_box_mat = np.stack(tmp_box_list, axis=0)
        img_box_list.append(tmp_box_mat)
    img_box_mat = np.stack(img_box_list, axis=0)
    for obj_id, obj_rgb in enumerate(obj_rgb_list):
        obj_rgb_list[obj_id] = obj_rgb / img_num 
    cost_matrix = np.zeros((obj_num, len(target_color_map))) 
    for obj_id, obj_val in enumerate(obj_rgb_list):
        for color_id, color_val in enumerate(target_color_map):
            tmp_diff = obj_val - color_val
            cost_matrix[obj_id, color_id] = np.sqrt(np.sum(tmp_diff**2))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    tmp_color_list = []
    for obj_idx in range(obj_num):
        obj_id = row_ind[obj_idx]
        color_id = col_ind[obj_idx]
        tmp_color_list.append(color_list[color_id])
    out_img_path = os.path.join(opts.visualize_dir, 
           str(vid)+'_' +str(img_id)+'_'+str(obj_id-1)+'.png')

    fall_ann_fn = os.path.join(opts.ann_dir, '../real.txt')
    video_fall_mat = np.loadtxt(fall_ann_fn)
    video_attr['trajectory'] = img_box_mat 
    video_attr['color'] = tmp_color_list
    video_attr['fall'] = bool(video_fall_mat[vid-1, -1])
    assert int(video_fall_mat[vid-1, 0])==vid

    vid_ann_path = os.path.join(opts.parse_annotation_folder, str(vid)+'.pkl')
    pickledump(vid_ann_path, video_attr)
    if opts.debug:
        visualize_specific_frame(img, tmp_box_mat, tmp_color_list, out_img_path, color_dict)

def pickledump(path, this_dic):
    f = open(path, 'wb')
    this_ans = pickle.dump(this_dic, f)
    f.close()

def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f)
    f.close()

def parse_video_info(opts, vid):
    img_fn = os.path.join(opts.ann_dir, 'img_concat_'+str(vid)+'.png')
    #mask_color_fn = os.path.join(opts.ann_dir, 'mask_order_'+str(vid)+'.png')
    mask_color_fn = os.path.join(opts.ann_dir, 'mask_color_'+str(vid)+'.png')
    mask_color_concat = cv2.imread(mask_color_fn)
    H_concat, W, C = mask_color_concat.shape
    img_concat = cv2.imread(img_fn)

    H = 224
    assert H_concat % H ==0
    assert H==W
    img_num = H_concat // H
    kernel = np.ones((5,5), np.uint8)
    video_attr = {}

    # BGR
    red_value = np.array((0,0, 255))
    yellow_value = np.array((0,255,255))
    green_value = np.array((0, 255, 0))
    blue_value = np.array((255, 0, 0))
    color_list = ['red', 'yellow', 'green', 'blue']
    target_color_map = [red_value, yellow_value, green_value, blue_value ]
    color_dict = {color:target_color_map[c_id] for c_id, color in enumerate(color_list)} 
        
    mask_color = mask_color_concat[:H] 
    obj_num = np.max(mask_color)
    obj_buffer_list =  [[] for idx in range(obj_num)]
    obj_rgb_list = [np.array([0, 0, 0], dtype=np.float64) for idx in range(obj_num)]

    img_box_list = []
    for img_id in range(img_num):
        st_row_id = img_id * H
        ed_row_id = img_id * H + H
        mask_color = mask_color_concat[st_row_id: ed_row_id] 
        img = img_concat[st_row_id: ed_row_id] 
        max_id = np.max(mask_color)
        tmp_box_list = []
        for obj_id in range(1, max_id+1):
            tmp_mask  = ((mask_color==obj_id)*255).astype(np.uint8)
            tmp_mask = cv2.morphologyEx(tmp_mask, cv2.MORPH_OPEN, kernel)
            ans = cv2.connectedComponentsWithStats(tmp_mask[:, :, 0])
            pdb.set_trace()

            mask = np.sum(tmp_mask, axis=2)  
            mask_x = np.sum(mask, axis=0)
            mask_y = np.sum(mask, axis=1)
            
            if np.sum(mask)==0:
                tmp_box = np.array([0, 0, 0, 0])
                tmp_box_list.append(tmp_box)
                continue 

            for x_id in range(W):
                if mask_x[x_id] >0:
                    x1 = x_id
                    break
            for x_id in range(W-1, 0, -1):
                if mask_x[x_id] >0:
                    x2 = x_id
                    break
            for y_id in range(H):
                if mask_y[y_id] >0:
                    y1 = y_id
                    break
            for y_id in range(H-1, 0, -1):
                if mask_y[y_id] >0:
                    y2 = y_id
                    break
            tmp_box = np.array([x1, y1, x2, y2])
            tmp_box_list.append(tmp_box)
            bi_mask  = tmp_mask == 255
            pos_pixel_num = np.sum(bi_mask[:, :, 0])
            mean_obj_rgb = 1/(0.000001+pos_pixel_num) * np.sum((img * bi_mask).reshape(-1, C), axis=0)
            obj_rgb_list[obj_id-1] +=mean_obj_rgb
            obj_buffer_list[obj_id-1].append(mean_obj_rgb)
            
            tmp_mask = cv2.rectangle(tmp_mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.imwrite('dumps/visualization/mask/'+str(img_id)+'_'+str(obj_id-1)+'.png', tmp_mask.astype(np.uint8))
            
            #pdb.set_trace()
        tmp_box_mat = np.stack(tmp_box_list, axis=0)
        img_box_list.append(tmp_box_mat)
    img_box_mat = np.stack(img_box_list, axis=0)
    video_attr['trajectory'] = img_box_mat 
    for obj_id, obj_rgb in enumerate(obj_rgb_list):
        obj_rgb_list[obj_id] = obj_rgb / img_num 
    cost_matrix = np.zeros((obj_num, len(target_color_map))) 
    for obj_id, obj_val in enumerate(obj_rgb_list):
        for color_id, color_val in enumerate(target_color_map):
            tmp_diff = obj_val - color_val
            cost_matrix[obj_id, color_id] = np.sqrt(np.sum(tmp_diff**2))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    tmp_color_list = []
    for obj_idx in range(obj_num):
        obj_id = row_ind[obj_idx]
        color_id = col_ind[obj_idx]
        tmp_color_list.append(color_list[color_id])
    out_img_path = os.path.join(opts.visualize_dir, 'boxes', 
           str(vid)+'_' +str(img_id)+'_'+str(obj_id-1)+'.png')
    visualize_specific_frame(img, tmp_box_mat, tmp_color_list, out_img_path, color_dict)
    #pdb.set_trace()
    


def visualize_specific_frame(img, box_mat, color_list, out_img_path, color_dict):
    obj_num = box_mat.shape[0]
    for obj_id in range(obj_num):
        tmp_box = box_mat[obj_id]
        color = color_list[obj_id]
        if color=='white':
            continue 
        color_info = color_dict[color].tolist() 
        x1, y1, x2, y2 = int(tmp_box[0]), int(tmp_box[1]), int(tmp_box[2])+1, int(tmp_box[3])+1    
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_info, 1)
        cv2.putText(img, color, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_info, 1)
    out_dir = os.path.dirname(out_img_path)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(out_img_path, img.astype(np.uint8))
