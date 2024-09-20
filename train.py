"""
Transformer torch reimplementation
Good refs:
- https://github.com/hkproj/pytorch-transformer/tree/main  
- https://github.com/AkiRusProd/numpy-transformer/blob/master/transformer/transformer.py

    ## Karpathy GPT 
- https://www.youtube.com/watch?v=kCc8FmEb1nY
- https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=Hs_E24uRE8kr


"""
# import torch.multiprocessing as mp
# if mp.get_start_method(allow_none=True) is None:
#     mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse, os, time, random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange 

import torch
from torchvision.transforms import transforms
import torchmetrics 
from torch.utils.data import Dataset, DataLoader, random_split 
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch import nn 

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel 
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# file imports 
import arguments
from models.transfomers.transformer import build_transformer
from dataset import BilingualDataset, causal_mask

from tqdm import tqdm, trange 
from pathlib import Path

from utils import TIMESTAMP

# torch.cuda.set_per_process_memory_fraction(0.7)

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def build_tokenizer(config, dataset, lang):
    """Build tokenizer from HuggingFace. Mapping each word to a number, seperated with whitespace."""
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    dataset_raw = load_dataset(f'{config["datasource"]}', f'{config["lang_src"]}-{config["lang_trg"]}', split='train')

    # build tokenizers
    tokenizer_src = build_tokenizer(config, dataset_raw, config["lang_src"])
    tokenizer_trg = build_tokenizer(config, dataset_raw, config["lang_trg"])

    # split validation set
    train_dataset_size = int(0.9 * len(dataset_raw))
    val_dataset_size = len(dataset_raw) - train_dataset_size

    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_dataset_size, val_dataset_size])
    
    
    train_dataset = BilingualDataset(train_dataset_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_trg'], config['seq_len'])
    val_dataset = BilingualDataset(val_dataset_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_trg'], config['seq_len'])


    maxlen_src, maxlen_trg = 0, 0

    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        trg_ids = tokenizer_trg.encode(item['translation'][config['lang_trg']]).ids
        maxlen_src = max(maxlen_src, len(src_ids))
        maxlen_trg = max(maxlen_trg, len(trg_ids))

    print(f'Max length of source sentence: {maxlen_src}')
    print(f'Max length of target sentence: {maxlen_trg}')

    train_dl = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dl, val_dl, tokenizer_src, tokenizer_trg

def get_model(config, vocab_src_len, vocab_trg_len):
    return build_transformer(
        src_vocab_size=vocab_src_len, 
        trg_vocab_size=vocab_trg_len, 
        src_seq_len=config['seq_len'], 
        trg_seq_len=config['seq_len'], 
        d_model=config['d_model']
    )

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_trg, maxlen, device):
    """Greedy search"""
    sos_idx = tokenizer_trg.token_to_id('[SOS]')
    eos_idx = tokenizer_trg.token_to_id('[EOS]')

    # Precompute the encoder output and reuse for every token from the decoder
    enc_output = model.encode(source, source_mask)
    
    dec_input = torch.empty(1,1).fill_(eos_idx).type_as(source).to(device)

    while True:
        if dec_input.size(1) == maxlen:
            break

        # build mask for target
        dec_mask = causal_mask(dec_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(enc_output, source_mask, dec_input, dec_mask)

        prob = model.project(out[:, -1])

        _, next_word = torch.max(prob, dim=1)

        dec_input = torch.cat(
            [dec_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], 
            dim=1
        )

        if next_word == eos_idx:
            break
            
    return dec_input.squeeze(0)


def eval(    
    model, val_dl, tokenizer_src, tokenizer_trg, maxlen, device, 
    print_msg, global_step, writer, num_examples=2
):
    """Validation loop"""
    model.eval()
    count = 0 

    src_texts = []
    targets, preds = [], []

    try:
        with os.open('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80
    
    with torch.no_grad():
        # batch_iter = tqdm(enumerate(val_dl), colour='red')
        # for batch in trange(val_dl, desc='Eval batch', colour='red'):
        for i, batch in enumerate(val_dl):
            count += 1
            enc_input = batch['encoder_input'].to(device)
            enc_mask = batch['encoder_mask'].to(device)

            assert enc_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, enc_input, enc_mask, tokenizer_src, tokenizer_trg, maxlen, device)

            src_text = batch['src_text'][0]
            trg_text = batch['trg_text'][0]

            model_out_text = tokenizer_trg.decode(model_out.detach().cpu().numpy())

            src_texts.append(src_text)
            targets.append(trg_text)
            preds.append(model_out_text)


            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{src_text}")
            print_msg(f"{f'TARGET: ':>12}{trg_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
            break
    if writer:
        # character error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(preds, targets)
        writer.add_scalar('validation CER', cer)
        writer.flush()
        
        # character error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(preds, targets)
        writer.add_scalar('validation WER', wer)
        writer.flush()
        
        # BLEU
        metric = torchmetrics.BLEUScore()
        bleu = metric(preds, targets)
        writer.add_scalar('validation BLEU', bleu)
        writer.flush()
        
    
def train(config):
    """Main training loop"""
    print(f'Using device: {device}')


    # TODO: change with create_folder()
    Path(f'{config["datasource"]}_{config["model_folder"]}').mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, tokenizer_src, tokenizer_trg = get_dataset(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size()).to(device)

    
    # ***** Tensorboard preparation *****
    writer = SummaryWriter(f'runs/{TIMESTAMP}')


    # ***** End of Tensorboard preparation *****

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)


    init_epoch = 0
    global_step = 0
    preload = config['preload']

    from utils import get_weights_filepath, latest_weights_filepath
    model_fn = latest_weights_filepath(config) if preload == 'latest' else get_weights_filepath(config, preload)

    if model_fn: 
        print(f'Preloading model {model_fn}')
        state = torch.load(model_fn)
        model.load_state_dict(state['mdoel_state_dict'])
        init_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])    
        global_step = state['global_step']

    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)

    epochs = config['epochs']
    for epoch in range(init_epoch, epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iter = tqdm(train_dl, desc=f'Epoch {epoch}', colour='green')

        for batch in batch_iter:
            enc_input = batch['encoder_input'].to(device)
            dec_input = batch['decoder_input'].to(device)
            enc_mask = batch['encoder_mask'].to(device)
            dec_mask = batch['decoder_mask'].to(device)


            enc_output = model.encode(enc_input, enc_mask)
            dec_output = model.decode(enc_output, enc_mask, dec_input, dec_mask)    
            proj_output = model.project(dec_output)
        
            label = batch['label'].to(device) # shape: (batch, seqlen)

            loss = loss_fn(proj_output.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1))
            batch_iter.set_postfix({'loss': f'{loss.item()}'})
            
            writer.add_scalar('Train loss', loss.item(), global_step)
            writer.flush()

            # NOTE: revert this based on best practice 
            # NOTE: Q: should we make this batch or stochastic like this
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        eval(model, val_dl, tokenizer_src, tokenizer_trg, config['seq_len'], device, 
                       lambda msg : batch_iter.write(msg), global_step, writer)
        
        model_fn = get_weights_filepath(config, f'{epoch:02d}')
        
        torch.save(
            {
                'epoch': epoch, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 
                'global_step': global_step, 
            }, 
            model_fn
        )
        
        
if __name__== '__main__':
    parser = argparse.ArgumentParser()
    arguments.add_common_args(parser)
    arguments.add_model_args(parser)
    args = parser.parse_args()

    print('-'*30); print(f'Config: {args}'); print('-'*30)
    # build_tokenizer(args)
    config = vars(args)
    train(config)
    