"""
Compilations of hyperparameters definitions for argparse   
"""
from pathlib import Path

create_folder = lambda path : os.makedirs(path, exist_ok=True) if not os.path.exists(path) else None


# import argparse
def add_common_args(parser):
    """Default arguments"""
    parser.add_argument('-tp', '--train_path', default=None)
    parser.add_argument('-vp', '--val_path', default=None)

    parser.add_argument('-pkl', '--data_pkl', default=None)

    parser.add_argument('-out', '--output_dir', type=str, default='./output')
    parser.add_argument('-tb' '--tensorboard', action='store_true')
    parser.add_argument('-save', '--save_mode', type=str, default='all', choices=['all', 'best'])
    parser.add_argument('-runs', '--runs', type=str, default=None)
    # reload checkpoints
    parser.add_argument('-reload', '--reload', type=str, default='latest')
    parser.add_argument('-tokfn', '--token_filename', type=str, default='tokenizer_0.json')

    # dataset related args
    parser.add_argument('-folder', '--model_folder', default='weights')
    parser.add_argument('-basename', '--model_basename', default='tmodel_')
    parser.add_argument('-ds', '--datasource', type=str, default='opus_books')
    parser.add_argument('-preload', '--preload', type=str, default='latest')
    parser.add_argument('-tokenizer_file', '--tokenizer_file', type=str, default='tokenizer_[0].json')
    parser.add_argument('-langsrc', '--lang_src', type=str, default='en')
    parser.add_argument('-langtrg', '--lang_trg', type=str, default='it')

def add_model_args(parser):
    # training related args
    parser.add_argument('-ep', '--epochs', type=int, default=20)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-seed', '--seed', type=int, default=None)
    parser.add_argument('-lr', '--lr', type=float, default=1e-4)
    
    # transformer related args 
    parser.add_argument('-d', '--d_model', type=int, default=512)
    parser.add_argument('-lr' '--lr', type=int, default=1e-3)
    parser.add_argument('-seqlen', '--seq_len', type=int, default=350)

    

def get_ckpt_path(config, epoch):
    model_folder = f'{config["datasource"]}_{config["model_folder"]}'
    model_filename = f'{config["model_basename"]}_ep{epoch}.pt'

    return str(Path('.') / model_folder / model_filename)

def latest_ckpt_path(config):
    model_folder = f'{config["datasource"]}_{config["model_folder"]}'
    ckpt_str_pattern = f'{config["model_basename"]}*'
    ckpt_files = list(Path(model_folder).glob(ckpt_str_pattern))
    if len(ckpt_files) == 0:
        return None
    ckpt_files.sort()
    return str(ckpt_files[-1])


    # parser.add_argument('-dih' '--d_inner_hid', type=int, default=2048)
    # parser.add_argument('-dk', '--d_k', type=int, default=64)
    # parser.add_argument('-dv', '--d_v', type=int, default=64)

    # parser.add_argument('-heads', '--n_heads', type=int, default=8)
    # parser.add_argument('-layers', '--n_layers', type=int, default=6)
    # parser.add_argument('-warmup', '--warmup_steps', type=int, default=4000)
    
    # parser.add_argument('-do', '--dropout', type=float, default=0.1)
    # parser.add_argument('-emb_sw', '--embs_shared_weight', action='store_true')
    # parser.add_argument('-proj_sw', '--proj_share_weight', action='store_true')
    # parser.add_argument('-scale', '--scale_emb_or_prj', type=str, default='prj')

    # parser.add_argument('-ls', '--label_smoothing', action='store_true')


        

