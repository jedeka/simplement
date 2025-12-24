import numpy as np
import torch

import os, subprocess, shutil, random
import json
import matplotlib.pyplot as plt


create_folder = lambda path : os.makedirs(path, exist_ok=True) if not os.path.exists(path) else None
get_params_num = lambda m : print(f"Params: {sum(p.numel() for p in m.parameters()):,}")

# GPU settings
def get_lowest_gpu(machine='lebniz', verbose=True):
    """[DELETE LATER] Get GPU with current lowest memory usage"""
    try:
        # Run nvidia-smi command to get GPU information
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output = result.stdout.strip()
        
        # print(output)
        # Parse GPU memory usage information
        gpu_memory_usages = [int(memory.strip()) for memory in output.split('\n')]
        
        if machine.lower() == 'leibniz': 
            # workaround to prevent using leibniz's gpu no.2
            gpu_memory_usages[2] = 99999 
        
        # Find GPU with lowest memory usage
        lowest_memory_index = gpu_memory_usages.index(min(gpu_memory_usages))

        if verbose:
            print(f'Using device: {lowest_memory_index}, with current memory {gpu_memory_usages[lowest_memory_index]} MB')

        return lowest_memory_index

    except Exception as e:
        print("Error:", e)
        print('Returning cuda:0.')
        return 0
    

def set_seed(seed, verbose=True):
    if seed == -1:
        seed = 0 
    if verbose:
        print('Setting SEED to', seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_bandit_data_filename(env, n_envs, config, mode):
    """Build filename for the bandit pretraining dataset"""
    raise NotImplementedError

# DEVICE_NO = get_lowest_gpu()  
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# experiment-related utils
from datetime import datetime # TODO: TIDY CODE LATER
from pytz import timezone

class Timestamp:
    """Class to process timestamp-related I/O task"""
    # def __init__(self, timestamp=None, timezone=None): # NOTE: Revert to this later after publishing
    def __init__(self, timestamp=None, tz=None, fmt=None):
        """
        tz: timezone; set to a string like `Asia/Taipei`
        """
        if timestamp is not None:
            self.timestamp = timestamp
        else:
            now = datetime.now() if tz is None else datetime.now(timezone(tz))
            datenow, timenow = now.strftime("%Y%m%d %H:%M:%S" if fmt is None else fmt).split(' ')
            
            # manual modify to Taiwan time 
            # datenow, timenow = datetime.now().strftime("%m%d %H:%M:%S").split(' ')
            # h = str((int(timenow[:2]) + 8) % 24)
            # timenow = '0' + h + timenow[2:] if len(h) == 1 else h + timenow[2:] 
            # print(datenow)
            self.datenow = datenow
            self.timenow = timenow
            
            self.timestamp = f'{datenow}-{timenow}'

    def get_timestamp(self):
        return self.timestamp
    

def testlog(*args, above=False, below=False, newline_cnt=1, end=' '):
    """custom logger helper function
    NOTE: DELETE LATER
    """
    import os, datetime
    if above: print('\n'*newline_cnt); print('*'*30)
    print(f"[{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | {os.path.basename(__file__)}]", end=end)
    for i, content in enumerate(args):
        print(content,end=' ') if i < len(args)-1 else print(content)
        
    if below: print('\n'); print('*'*30); print('\n'*newline_cnt)

def convert_to_tensor(x, device, store_gpu=True):
    if store_gpu:
        return torch.tensor(np.asarray(x)).float().to(device)
    else:
        return torch.tensor(np.asarray(x)).float()

def save_to_pkl(trajs:dict, fn:str):
    with open(fn, 'wb') as f:
        print(f'Saving trajectories to {fn}')
        pickle.dump(trajs, f)

def load_from_pkl(fn:str): 
    trajs = [] 
    with open(fn, 'rb') as f:
        trajs.append(pickle.load(f))
    return trajs 


def plot(plots, xlabel='x', ylabel='y', title='plot', legend=True, save=None):
    """Take plots as dictionary"""
    # plt.figure(figsize=(10, 6))
    for key, value in plots.items():
        plt.plot(value, label=key)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if legend: plt.legend(shadow=True)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False, shadow=True, ncol=5)

    if save is not None:
        plt.savefig(save)
    

def plot_ax(ax, plots, xlabel='x', ylabel='y', title='plot', legend=True, save=None):
    """Take plots as dictionary"""
    # plt.figure(figsize=(10, 6))
    for key, value in plots.items():
        ax.plot(value, label=key)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)
    # if legend: ax.legend(shadow=True)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False, shadow=True, ncol=5)

    # if save is not None:
    #     ax.savefig(save)
    
    # plt.close()

def pplot(plots, xlabel='x', ylabel='y', title='plot', save=None, cpt=None):
    """Take plots as dictionary"""
    plt.figure(figsize=(10, 6))
    for key, value in plots.items():
        plt.plot(value, label=key)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if cpt is not None:
        for change_pt in cpt: plt.axvline(x=change_pt, ls=':', color='grey', alpha=0.5)
    plt.title(title)
    plt.legend(shadow=True)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False, shadow=True, ncol=5)

    if save is not None:
        plt.savefig(save)

    # plt.close()

# dataname 
def old_set_dataset_filename(cfg, mode, main_dataset_dir=None, postfix=None):
    """Just for testing. Perhaps better this one?"""
    main_dataset_dir = 'zdatasets' if main_dataset_dir is None else main_dataset_dir
    create_folder(main_dataset_dir)
    envcfg, datasetcfg = cfg['env'], cfg['dataset']


    filename = f'{main_dataset_dir}/trajs'
    filename += f'_{cfg["env"]["type"]}'
    filename += f'_arm{cfg["env"]["n_arms"]}'
    filename += f'_hor{cfg["env"]["horizon"]}'
    if mode != 'eval':
        filename += f'_envs{cfg["env"]["n_envs"]}'
    else:
        filename += f'_envs{cfg["env"]["n_eval_envs"]}'
    # print(envcfg.keys())
    drift_type, cp = cfg["env"]["drift_type"], cfg["env"]["drift"]["prob"]
    filename += f'_drift-{drift_type}'
    filename += f'_cp{cp}'

    filename += f'_{mode}'
    
    if envcfg["extreme"]:
        filename += '_extreme'
    if envcfg["optimal"]:
        filename += '_optimal'
    if envcfg["expert"]:
        filename += '_expert'
        filename += f'-{datasetcfg["expert"]["expert_type"]}'
        filename += f'-e{datasetcfg["expert"]["portion"]}'
        filename += f'n{datasetcfg["normal"]["portion"]}'
    
    # if 'postfix' in cfg.keys():
    if postfix is not None:
        filename += postfix
    
    filename += '.pkl'
    print('Setting data filename:', filename)
    return filename
    

def old_set_model_filename(cfg, postfix=None):
    """Set model checkpoint filename template
    NOTE: Set postfix outside.
    """
    main_models_dir = 'zmodels' # NOTE: delete later
    create_folder(main_models_dir)
    envcfg, datasetcfg = cfg['env'], cfg['dataset']

    env = cfg['env']['type']
    filename = env 
    filename += f'_arm{cfg["env"]["n_arms"]}'
    filename += f'_hor{cfg["env"]["horizon"]}'
    filename += f'_lr{cfg["train"]["lr"]}'
    filename += f'_envs{cfg["env"]["n_envs"]}'
    filename += f'_do{cfg["train"]["dropout"]}'
    filename += f'_shuf{cfg["shuffle"]}'

    if 'cp' in cfg.keys() and env == 'nsbandit':
        drift_type, cp = cfg["env"]["drift_type"], cfg["env"]["drift"]["prob"]
        filename += f'_drift-{drift_type}'
        filename += f'_cp{cp}'
    if envcfg["extreme"]:
        filename += '_extreme'
    if envcfg["expert"]:
        filename += '_expert'
        filename += f'-{datasetcfg["expert"]["expert_type"]}'
        filename += f'-e{datasetcfg["expert"]["portion"]}'
        filename += f'n{datasetcfg["normal"]["portion"]}'

    filename += f'_seed{cfg["seed"]}'

    if postfix is not None:
        filename += postfix
        
    print('Setting model filename:', filename)
    return filename



# dataname 
def set_dataset_filename(cfg, mode, main_dataset_dir=None, prefix='', postfix=''):
    """Just for testing. Perhaps better this one?"""
    main_dataset_dir = 'datasets' if main_dataset_dir is None else main_dataset_dir
    # create_folder(main_dataset_dir)
    envcfg, datasetcfg = cfg['env'], cfg['dataset']
    env_type = cfg['env']['type']

    filename = f'{main_dataset_dir}/{prefix}trajs'
    filename += f'_{cfg["env"]["type"]}'
    filename += f'_arm{cfg["env"]["n_arms"]}'
    filename += f'_hor{cfg["env"]["horizon"]}'
    if mode != 'eval':
        filename += f'_envs{cfg["dataset"]["n_envs"]}'
    else:
        filename += f'_envs{cfg["dataset"]["n_eval_envs"]}'

    # nsbandit properties
    if env_type == 'nsbandit':
        filename += f'_var{cfg["env"]["var_drift"]}'
        drift_mode = cfg["env"]["drift_mode"]
        if drift_mode == 'periodic': 
            drift_period = cfg["env"]["cfg_periodic"]["drift_period"]
            drift_mode += str(drift_period)
        drift_type = cfg["env"]["drift_type"]

        filename += f'_drift-{drift_mode}-{drift_type}'
        ## change/drift probability threshold
        cp = cfg["env"]["drift_prob"]
        filename += f'_thres{cp}'

    filename += f'_{mode}'
    
    if datasetcfg["extreme"]:
        filename += f'_extreme'
    if datasetcfg["optimal"]:
        filename += f'_optimal'
    if datasetcfg["expert"]:
        filename += '_expert'
        expert_name = datasetcfg["cfg_expert"]["expert_type"]
        if 'window_size' in datasetcfg["cfg_expert"].keys():
            window_size = datasetcfg["cfg_expert"]['window_size']
            expert_name = f'{window_size}{expert_name}'
        filename += f'-{expert_name}'
        filename += f'-e{datasetcfg["cfg_expert"]["portion"]}'
        filename += f'n{datasetcfg["cfg_base"]["portion"]}'
    
    # if 'postfix' in cfg.keys():
    # if postfix is not None:
    filename += postfix
    filename += '.pkl'

    # recursively create fo lder along with the prefix if prefix contains directory path
    create_folder('/'.join(filename.split('/')[:-1]))

    print('Setting data filename:', filename)
    return filename
    

def set_model_filename(cfg, postfix=None):
    """Set model checkpoint filename template
    NOTE: Set postfix outside.
    """
    main_models_dir = 'zmodels' # NOTE: delete later
    create_folder(main_models_dir)
    envcfg, datasetcfg = cfg['env'], cfg['dataset']

    env = cfg['env']['type']
    filename = env 
    filename += f'_arm{cfg["env"]["n_arms"]}'
    filename += f'_hor{cfg["env"]["horizon"]}'
    filename += f'_lr{cfg["train"]["lr"]}'
    filename += f'_envs{cfg["dataset"]["n_envs"]}'
    filename += f'_do{cfg["train"]["dropout"]}'
    filename += f'_shuf{cfg["shuffle"]}'

    if 'cp' in cfg.keys() and env == 'nsbandit':
        drift_type, cp = cfg["env"]["drift_type"], cfg["env"]["drift"]["prob"]
        filename += f'_drift-{drift_type}'
        filename += f'_cp{cp}'
    if datasetcfg["extreme"]:
        filename += '_extreme'
    if datasetcfg["expert"]:
        filename += '_expert'
        expert_name = datasetcfg["cfg_expert"]["expert_type"]
        if 'window_size' in datasetcfg["cfg_expert"].keys():
            window_size = datasetcfg["cfg_expert"]['window_size']
            expert_name = f'{window_size}{expert_name}'
        filename += f'-{expert_name}'
        filename += f'-e{datasetcfg["cfg_expert"]["portion"]}'
        filename += f'n{datasetcfg["cfg_base"]["portion"]}'
    
    filename += f'_seed{cfg["seed"]}'

    if postfix is not None:
        filename += postfix
        
    print('Setting model filename:', filename)
    return filename



import pickle
import numpy as np

def readpkl(path):
    l = []
    with open(path, 'rb') as f:
        while True:
            try:
                l.append(pickle.load(f))
            except EOFError:
                break
        return l

def copydir(src, dst): shutil.copytree(src=src, dst=dst)
def copyfile(src, dst): shutil.copyfile(src=src, dst=dst)



# current timestamp for file logging 
TIMESTAMP = Timestamp(tz='Asia/Taipei').timestamp

# CUDA device used 
DEVICE = get_lowest_gpu()  
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

# DEVICE = get_lowest_gpu(machine='whatever')  
# device = torch.device(f'cuda:{DEVICE}' if torch.cuda.is_available() else 'cpu')


# def plot_with_drifts(plots, xlabel='x', ylabel='y', title='plot', xticks=None, yticks=None, save=None, chpts=None):
#     """Plot with drifts"""
#     plt.figure(figsize=(10, 6))
#     for key, value in plots.items():
#         plt.plot(value, label=key)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)

#     if xticks is not None:
#         plt.xticks(xticks)
    
#     if yticks is not None:
#         plt.yticks(yticks)

def plot_with_drifts(plots, xlabel='x', ylabel='y', title='plot', xticks=None, yticks=None, legend=False, save=None, chpts=None):
    """Plot with drifts"""
    # plt.figure(figsize=(10, 6))
    for key, value in plots.items():
        plt.plot(value, label=key)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xticks is not None:
        plt.xticks(xticks)
    
    if yticks is not None:
        plt.yticks(yticks)
    # plt.grid(visible=True, which="major", color="lightgray", linestyle="--")
    plt.title(title)
    if legend: plt.legend(shadow=True)
    if chpts is not None:
        for change_pt in chpts:
            plt.axvline(x=change_pt, ls=':', color='grey', alpha=0.5)
    if save is not None: 
        plt.savefig(save)

def plot_ax_with_drifts(ax, plots, xlabel='x', ylabel='y', title='plot', xticks=None, yticks=None, marker=None, legend=False, chpts=None):
    """Plot with drifts"""
    # plt.figure(figsize=(10, 6))
    for key, value in plots.items():
        ax.plot(value, label=key, marker=marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)

    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if legend: plt.legend(shadow=True)
    if chpts is not None:
        for change_pt in chpts:
            ax.axvline(x=change_pt, ls=':', color='grey', alpha=0.5)

    # plt.title(title)
    # plt.legend(shadow=True)
    # if chpts is not None:
    #     for change_pt in chpts:
    #         plt.axvline(x=change_pt, ls=':', color='grey', alpha=0.5)
    # if save is not None: 
    #     plt.savefig(save)

def compare_model_parameters(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(param1, param2):
            print('*'*40); print('Different weights:')
            print(param1)
            print('x'*40)
            print(param2)
            print('*'*40)

            return False
    return True

def clear_runs():
    import glob 
    import shutil 

    recs_ = glob.glob('records/*')
    runs_ = glob.glob('runs/*')

    clean_recs = [path.split('/')[1] for path in recs_]
    delete_list = []

    for i, r in enumerate(runs_.copy()):
        path = r.split('/')[1].split('_')[0]
        if path not in clean_recs:
            print(f'path {i}->{path} is not in clean records. deleting')
            delete_list.append(r)

    # clean runs 
    for p in delete_list:
        try:
            print(f'Deleting {p}')
            shutil.rmtree(p)
        except FileNotFoundError:
            print(f'File {p} not found, skipping.')
            continue
        except OSError:
            print('OSError encountered, retrying to clear folders by ignoring the errors')
            shutil.rmtree(p, ignore_errors=True)
    print('All clean.')

def save_model(model, config, fn):
    torch.save(
        {'model_state_dict': model.state_dict(), 'config': config}, 
        fn
    )

def printdict(d):
    import json
    print(json.dumps(d, indent=2))

def rprint(x, prefix='', print_content=False):
    """Helper to recursively print dict"""
    if isinstance(x, dict):
        for k, v in x.items():
            print(prefix + k)
            rprint(v, prefix+'   ')
    elif type(x) in [np.ndarray, torch.Tensor, list]:
        if print_content:
            print(f'{prefix}{x}')
        else:
            print(f'{prefix}{type(x)} : {x.shape if type(x) is not list else len(x)}')
    else:
        print(f'{prefix}{x}')


class Logger:
    """Simple logging utility
    TODO: 
    - add summary writer 
    """
    # declare logging levels
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    DEBUG = 'DEBUG'

    def __init__(self, filename=None, console:bool=False, log2file:bool=True, print_time=True):
        # for directory creation
        dirs = filename.split('/')[:-1]
        self.dir = ''#.join(dirs) 
        if len(dirs) > 0:
            for dir in dirs:
                self.dir += f'{dir}/'
            self.dir = self.dir[:-1] # trim last slash

        self.log_filename = filename if filename is not None else 'out.log'

        self.console = console
        self.ts = Timestamp(tz='Asia/Taipei')
        self.level = None
        self.print_time = print_time
        self.log2file = log2file

    def create_file(self):
        create_folder(self.dir) 
        if not os.path.isfile(self.log_filename):
            with open(self.log_filename, 'w') as f:
                pass
    
    def __call__(self, string):
        self.create_file()
        with open(self.log_filename, 'a') as f:
            # configure logging level string
            if self.level is None: level_str = ''
            elif self.level == self.WARNING: level_str = f'[{self.level}] '
            else: level_str = f'[{self.level}] '
            level_str = level_str if level_str != '' else ' '
            
            # configure timestamp
            timestamp = f'[{self.ts.get_timestamp()}]' 
            if not self.print_time:
                timestamp = ''
            # the whole string
            s = timestamp + level_str + string
            # post processing if string's leftside is just whitespaces 
            if timestamp + level_str == ' ': s = s.lstrip() 
            # print to file and console
            if self.log2file: print(s, file=f)
            if self.console: print(s) 
        self.level = None

    def info(self, string):
        # self.create_file()
        self.level = self.INFO 
        self(string)
    
    def warning(self, string):
        # self.create_file()
        self.level = self.WARNING
        self(string)

    def error(self, string):
        # self.create_file()
        self.level = self.ERROR
        self(string)
    
    def debug(self, string):
        # self.create_file()
        self.level = self.DEBUG
        self(string)
    

class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self. count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

import yaml 
def load_cfg(fn):
    with open(fn, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg 

def load_cfgs(dict_fn):
    cfgs = {}
    for k, fn in dict_fn.items():
        cfgs[k] = load_cfg(fn)
    return cfgs    

def vis(trajs, num=10, k=5, from_data=False): 
    """Visualize multiple trajectories
    NOTE:
    from_data: `True` if visualizing from dataset/dataloader
         (visualization before training), `False` if from 
         info dict after running evaluation.
    """
    # plt.figure(figsize=(20, 10))
    n = num//5+1 if num >= 5 else num
    if num < 5: n = num 
    elif num % 5 == 0: n = num // 5
    else: n = num//5 + 1
    m = 5 if num >= 5 else 1
    
    fig, axs = plt.subplots(n, m, figsize=(20, 10))  
    fig.suptitle('all_means`')
    if num > 1: axs = axs.ravel()
    for idx in range(num):
        if not from_data:
            chpts = trajs['chpts']
        # if from_data:
        #     chpts = np.where(chpts == True)[0]
            all_means = np.array(trajs['all_means'])
            a = { f'arm{i}': all_means[idx][:,i] for i in range(k) }
            _chpts = chpts[idx]
            # print(_chpts)
        else:
            chpts = trajs[idx]['change_points']
            chpts = np.where(chpts == True)[0]
            all_means = trajs[idx]['means']
            a = { f'arm{i}': all_means[:,i] for i in range(k)}
            _chpts = chpts
        
        if num > 1:
            plot_ax_with_drifts(axs[idx], a, chpts=_chpts)
        else:
            plot_with_drifts(a, chpts=_chpts)
    if num > 1:
        handles, labels = axs[0].get_legend_handles_labels() 
    else:
        handles, labels = axs.get_legend_handles_labels() 
        
    if k <= 10:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels), shadow=True)
    plt.tight_layout()


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
    # parser.add_argument('dataset', choices=('8gaussians', '2spirals', 'checkerboard', 'rings', 'MNIST'))
    # parser.add_argument('model', choices=('FCNet', 'ConvNet'))
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate. default: 1e-3')
    # parser.add_argument('--stepsize', type=float, default=0.1, help='Langevin dynamics step size. default 0.1')
    # parser.add_argument('--n_step', type=int, default=100, help='The number of Langevin dynamics steps. default 100')
    # parser.add_argument('--n_epoch', type=int, default=100, help='The number of training epoches. default 100')
    # parser.add_argument('--alpha', type=float, default=1., help='Regularizer coefficient. default 100')
    # args = parser.parse_args()
    

    # parser = argparse.ArgumentParser()
    # parser.add_argument('energy_function', help='select toy energy function to generate sample from. (u1, u2, u3, u4)',
    #                     choices=['u1', 'u2', 'u3', 'u4'])
    # parser.add_argument('--no-arrow', action='store_true', help='disable display of arrows')
    # parser.add_argument('--out', help='the name of output file. default is the name of energy function.  ex) u1.gif',
    #                     default=None)
    # args = parser.parse_args()

    # import json 
    # print(json.dumps(vars(args), indent=4))