import json
from modules.opts import options 
from modules.video_parser import Simulation 
from utils.util import parse_video_info, set_debugger, parse_video_info_v2  

set_debugger()

def parse_video_annotation(opts):
    ignore_list = []
    ignore_list += list(range(84, 89))
    ignore_list += list(range(345, 358))
    ignore_list += list(range(367, 371))
    for vid in range(1, 517):
        parse_video_info_v2(opts, vid)

if __name__=='__main__':
    option = options()
    opts = option.parse()
    
    if opts.parse_annotation_flag==1:
        parse_video_annotation(opts)
