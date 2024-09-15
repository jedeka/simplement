from typing import Any
import torch
from torch import nn 
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_trg, src_lang, trg_lang, seq_len):
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_trg.token_to_id('[SOS]')], dtype=torch.int64) # start of string
        self.eos_token = torch.tensor([tokenizer_trg.token_to_id('[EOS]')], dtype=torch.int64) # end of string
        self.pad_token = torch.tensor([tokenizer_trg.token_to_id('[PAD]')], dtype=torch.int64) # padding


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> Any:
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang]
        trg_text = src_target_pair['translation'][self.trg_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_trg.encode(trg_text).ids

        # important to maintain the same sequence length
        ## enc - 2 because pad SOS & EOS
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        ## dec - 1 because add SOS only
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        # add <s> and </s> token 
        enc_input = torch.cat([
            self.sos_token, 
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token, 
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ], dim=0)
        
        # add only <s> token
        dec_input = torch.cat([
            self.sos_token, 
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        # add only </s> token
        label = torch.cat([ 
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token, 
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ], dim=0)


        
        assert enc_input.size(0) == self.seq_len
        assert dec_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': enc_input, 
            'decoder_input': dec_input, 
            'encoder_mask': (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # shape: 1, 1,seq_len (adding batch & seq_dim)
            'decoder_mask': (enc_input != self.pad_token).unsqueeze(0).int() & causal_mask(dec_input.size(0)), # shape: 1, 1,seq_len & (1, seq_len, seq_len)
            'label': label, 
            'src_text': src_text, 
            "trg_text": trg_text 
        }
        

def causal_mask(size):
    """Creating causal mask for attention map.
    (1:48:00) Used since we want the each word from decoder to watch the previous words up to itself, NOT after.
    Attention map comes from Q @ K, it produces (d_model, d_model) map. 
    We want to mask the upper triangular part of the matrix. 
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    # since torch.triu only returns upper trig. of the mtx, but we want to 
    #   actually mask the upper trig., we return the reverse, where 
    #   the lower trig. mtx with value 0 will be 1. 
    return mask == 0 


