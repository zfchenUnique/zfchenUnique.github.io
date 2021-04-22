import os
import argparse


class options():
    """Expression generation options"""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--ann_dir', type=str, default='../real_blocks/block_camera_combined_scaled2')
        self.parser.add_argument('--ques_save_dir', type=str, default='dump/questions')
        self.parser.add_argument('--set_name', type=str, default='train')
        self.parser.add_argument('--visualize_dir', type=str, default='../clevrer/billiards/visualization/train')
        self.parser.add_argument('--generate_ques_flag', type=int, default=0)
        self.parser.add_argument('--standard_length', type=int, default=128)
        self.parser.add_argument('--split_video_flag', type=int, default=0)
        self.parser.add_argument('--set_file', type=str, default='../clevrer/billiards/train.txt')
        self.parser.add_argument('--visualize_video_flag', type=int, default=0)
        self.parser.add_argument('--debug', type=int, default=0)
        self.parser.add_argument('--balance_flag', type=int, default=0)
        self.parser.add_argument('--max_ques_num', type=int, default=16)
        self.parser.add_argument('--ques_refine_save_dir', type=str, default='dump/questions')
        self.parser.add_argument('--parse_annotation_flag', type=int, default=0)
        self.parser.add_argument('--parse_annotation_folder', type=str, default='../real_blocks/video_annotations')

    def parse(self):
        self.opt = self.parser.parse_args()

        args = vars(self.opt)
        return self.opt
