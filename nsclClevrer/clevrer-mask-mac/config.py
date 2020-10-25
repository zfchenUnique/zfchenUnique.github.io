import configparser
config = configparser.ConfigParser()
#config['epxeriment_path'] = './'
config['DEFAULT']={'experiment_path': './',
        'n_epoch': 30,
        'batch_size': 16,
        'num_workers': 2,
        'resume': '',
        'use_attr_flag': False,
        'type': 'open_ended',
        'video_features_path': './dump/blocks_video_feature.npy',
        'video_info_path': './dump/blocks_video_feature.pkl' ,
        'object_features_path': '' ,
        'video_split_info_path': '../question_blocks/dump/questions' ,
        'log_every_iter': 30
        }
config['pickle'] ={}
config['pickle']['question'] = './data/open_ended/question_dictionary.pkl'
config['pickle']['answer'] = './data/open_ended/answer_dictionary.pkl'
config['pickle']['family'] = './data/open_ended/family_dictionary.pkl'

config['model'] = {}
config['model']['arch'] = 'mac'

config['mac'] = {}
config.set('mac','in_channels', '1024' )
config['mac']['dim'] = '512'
config['mac']['net_length'] = '8' 
config['mac']['embedding_dim'] = '300'
config['mac']['self_attention'] ='True' 
config['mac']['memory_gate'] ='True' 
config['mac']['dropout'] = '0.15'
config['mac']['glove_path'] ='' 
