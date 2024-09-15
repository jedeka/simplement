from datetime import datetime # TODO: TIDY CODE LATER
import os
import torch
from pathlib import Path
import numpy as np

create_folder = lambda path : os.makedirs(path, exist_ok=True) if not os.path.exists(path) else None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def testlog(*args, above=False, below=False, newline_cnt=1):
    """custom logger helper function
    NOTE: DELETE LATER
    """
    import os, datetime
    if above: print('\n'*newline_cnt); print('*'*30)
    print(f"[{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | {os.path.basename(__file__)}]")#, end=' ')
    for i, content in enumerate(args):
        if i < len(args)-1: print(content,end=' ')
        else: print(content)
    if below: print('\n'); print('*'*30); print('\n'*newline_cnt)

def convert_to_tensor(x, store_gpu=True):
    if store_gpu:
        return torch.tensor(np.asarray(x)).float().to(device)
    else:
        return torch.tensor(np.asarray(x)).float()


class Timestamp:
    """Class to process timestamp-related I/O task"""
    # def __init__(self, timestamp=None, timezone=None): # NOTE: Revert to this later after publishing
    def __init__(self, timestamp=None, tz=None):
        """
        tz: timezone; set to a string like `Asia/Taipei`
        """
        if timestamp is not None:
            self.timestamp = timestamp
        else:
            from pytz import timezone

            now = datetime.now() if tz is None else datetime.now(timezone(tz))
            datenow, timenow = now.strftime("%m%d %H:%M:%S").split(' ')
            self.timestamp = f'{datenow}-{timenow}'

    def get_timestamp(self):
        return self.timestamp

def get_weights_filepath(config, epoch):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_fn = f"{config['model_basename']}{str(epoch)}.pt"
    # return str(Path('.') / model_folder / model_fn)
    return f'./{model_folder}/{model_fn}'

def latest_weights_filepath(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_fn = f"{config['model_basename']}"
    weights_files = list(Path(model_folder).glob(model_fn))

    if len(weights_files) == 0:
        return None 
    weights_files.sort()
    return str(weights_files[-1])

def get_weights_filename(config, epoch=None):
    """Get weights file path"""
    pass


# DEVICE = get_lowest_gpu(machine='leibniz')  
DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')


# current timestamp for file logging 
TIMESTAMP = Timestamp(tz='Asia/Taipei').get_timestamp()

