import json
import glob
import os
import argparse
import pycocotools.mask as _mask
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt



def decode(rleObjs):
    if isinstance(rleObjs, list):
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:, :, 0]


if __name__ == '__main__':
    """
    {
  "video_index": 5,
  "video_name": "sim_00005",
  "frames": [
    {
      "frame_filename": "sim_00005/frame_00000.png",
      "frame_index": 0,
      "objects": [
        {
          "mask": {
            "size": [
              320,
              480
            ],
            "counts": "jVc0`0^95K3N3M2M3N2N2O1N101N2N101N1000000000000000000000000000000000000001N101N2N101N2N2N2N2N3L3N3LYSd3"
          },
          "video_index": 5,
          "frame_index": 0,
          "frame_filename": "sim_00005/frame_00000.png",
          "color": "blue",
          "material": "metal",
          "shape": "sphere",
          "score": 0.9997047781944275
        },
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir",
                        default="/data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/arxiv_dec_2018/data/derender/processed_proposals")
    parser.add_argument("-o", "--output_dir",
                        default="/data/vision/torralba/scratch2/chuang/tgif_qa/resnet152-pynvvl/mask_all_object")

    # parser.add_argument("-i", "--input_dir", default="/home/xiaohongdong/TVQA")
    # parser.add_argument("-o", "--output_dir", default="/home/xiaohongdong/TVQA/mask")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    files = glob.glob(os.path.join(args.input_dir, "*.json"))
    sorted(files)
    attr = {"color": [], "material": [], "shape": []}

    for file in files:
        with open(file) as f:
            dic_ori = json.load(f)
        f.close()
        for i in range(len(dic_ori["frames"]) // 5):
            for j in range(len(dic_ori["frames"][i * 5]["objects"])):
                object_ = dic_ori["frames"][i * 5]["objects"][j]
                if object_["color"] not in attr["color"]:
                    attr["color"].append(object_["color"])
                if object_["material"] not in attr["material"]:
                    attr["material"].append(object_["material"])
                if object_["shape"] not in attr["shape"]:
                    attr["shape"].append(object_["shape"])
    fp = np.lib.format.open_memmap(
        os.path.join(args.output_dir, "mask_14x14.npy"), mode='w+', dtype=np.float32, shape=(20000, 25, 3, 14, 14))

    for file_index, file in enumerate(files):
        with open(file) as f:
            dic_ori = json.load(f)
        f.close()
        video_mask = []
        for i in range(len(dic_ori["frames"]) // 5):
            mask_final = np.zeros([3, 320, 480], dtype=np.float32)
            for j in range(len(dic_ori["frames"][i * 5]["objects"])):
                object_ = dic_ori["frames"][i * 5]["objects"][j]
                object_mask = decode(object_["mask"])
                # mask_final += object_mask
                # mask_final[0] += object_mask * (attr["color"].index(object_["color"])/color_max - 0.5)
                # mask_final[1] += object_mask * (attr["material"].index(object_["material"])/material_max - 0.5)
                # mask_final[2] += object_mask * (attr["material"].index(object_["material"])/shape_max - 0.5)

                mask_final[0] += object_mask * (attr["color"].index(object_["color"]) + 1) / len(attr["color"])
                mask_final[1] += object_mask * (attr["material"].index(object_["material"]) + 1) / len(attr["material"])
                mask_final[2] += object_mask * (attr["shape"].index(object_["shape"]) + 1) / len(attr["shape"])

            video_mask.append(mask_final)
        video_mask = np.asarray(video_mask).astype(np.float)
        video_mask = torch.from_numpy(video_mask)
        # for i in range(video_mask.shape[0]):
        #     plt.figure(figsize=(6, 4))
        #     plt.imshow(((video_mask[i].numpy() + 0.5).transpose((1, 2, 0)) * 255).astype(np.uint8), aspect="auto", origin="lower", interpolation='bilinear')
        #     plt.colorbar()
        #     plt.savefig(os.path.join(args.output_dir, dic_ori["video_name"] + "_" + str(i) + ".jpg"))
        #     plt.close()
        # os.system("/home/xiaohongdong/miniconda3/envs/py37/bin/ffmpeg  -f image2 -r 5 -i {} {}".format(
        #     os.path.join(args.output_dir, dic_ori["video_name"] + "_%d.jpg"),
        #     os.path.join(args.output_dir, dic_ori["video_name"] + ".mp4")
        # ))
        video_mask = F.interpolate(video_mask, size=(14, 14), mode="bilinear", align_corners=False)
        fp[file_index] = video_mask.numpy()
        print('index', file_index)
    del fp
