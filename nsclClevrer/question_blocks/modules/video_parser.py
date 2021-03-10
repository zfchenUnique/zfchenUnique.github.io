import json

class Simulation():
    """Interface for simulation data"""
    def __init__(self, opts, video_index):
        ann_full_path = os.path.join(opts.ann_dir, vid_str+'.pkl') 
