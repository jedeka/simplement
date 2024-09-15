import torch 
from torch import nn 
import torch.nn.functional as F
import math 

class FeedForwardBlock(nn.Module):
    """Feedforward block"""
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.Dropout(dropout), 
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        # shape: (batch, seq_len, d_model) -> (batch, seq_len, d_ff) ->  (batch, seq_len, d_model)
        return self.net(x)

class InputEmbedding(nn.Module):
    """Embedding layers, @paper section 3.4"""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        # @paper: multiply those weights by √d_model
        return self.embedding(x) * math.sqrt(self.d_model)        

class PositionalEncoding(nn.Module):
    """Positional encoding, @paper sec. 3.5
    * PE_(pos, 2i) = sin(pos/(10000)^(2i/d_model))
    * PE_(pos, 2i+1) = cos(pos/(10000)^(2i/d_model))
    """
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        
        # TODO: why this is shaped this way? 
        # numerator
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # denominator 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # wave functions
        pe[:, 0::2] = torch.sin(pos * div_term) # even indexes
        pe[:, 1::2] = torch.cos(pos * div_term) # odd indexes
        pe = pe.unsqueeze(0)

        # register_buffer() to include non-learnable tensor inside the module during saving 
        self.register_buffer('pe', pe)
    

    def forward(self, x):
        x += (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        x = self.dropout(x)

        return x 

class LayerNorm(nn.Module):
    """Layer normalization
    Refs:
    * https://stackoverflow.com/questions/70065235/understanding-torch-nn-layernorm-in-nlp
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.eps = eps 
        # alpha and bias are learnable params
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class ResidualConnection(nn.Module):
    def __init__(self, features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttention(nn.Module):
    """Multi head attention block, @paper sec. 3.2"""
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h 
        
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        
        # for learned linear projections for q, k, v
        self.W_q = nn.Linear(d_model, d_model)  
        self.W_k = nn.Linear(d_model, d_model)  
        self.W_v = nn.Linear(d_model, d_model) 
    
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    # @staticmethod to make decouple it from the instance (no need of instantiation), like a standalone function 
    @staticmethod
    # def scaled_dot_product_attention(query, key, value, mask, dropout:nn.Dropout):
    def attention(query, key, value, mask, dropout:nn.Dropout):
        """Scaled Dot-Product Attention @paper sec. 3.2.1
        * Consists of q & k of dimension d_k, v of dimension d_v.
        * Compute the dot products of q with all keys, divide each by sqrt(d_k)
        * Apply softmax function to obtain the weights on the values
        
        TODO: just put it outside perhaps?
        """
        d_k = query.shape[-1] # last dim of the query 
        
        # shape: (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores    
    
    def forward(self, q, k, v, mask):
        """
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
            where head_i = Attention(Q @ WQ_i, K @ WK_i, V @ WV_i)
        """
        
        # size: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        query = self.W_q(q) 
        key = self.W_k(k) 
        value = self.W_v(v) 

        # NOTE: can actually consider reshape and permute, but whatever
        # shape: (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        
        x, self.attention_scores = \
            MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # concat head_1, ..., head_h
        # shape: (batch, h, seq_len, d_v) -> (batch, seq_len, h, d_v) -> (batch, seq_len, d_model)
        # NOTE: in the paper, d_k == d_v == d_model, so we just use d_k here 
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # multiply by Wo, projection matrices
        # shape: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.W_o(x)
        
class EncoderBlock(nn.Module):
    """The encoder block"""
    def __init__(
        self, 
        features: int, 
        self_attn_block: MultiHeadAttention, 
        ff_block: FeedForwardBlock, 
        dropout: float
    ):
        super().__init__()
        self.self_attn_block = self_attn_block # self-attention (MHA) block
        self.ff_block = ff_block # feedforward block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attn_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.ff_block)
        return x
    
class Encoder(nn.Module):
    """Whole encoder"""
    def __init__(self, features, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, mask):
        """Add & Norm"""
        for layer in self.layers:
            x = layer(x, mask) 
        return self.norm(x)

class DecoderBlock(nn.Module):
    """The decoder block"""
    def __init__(
        self, 
        features: int, 
        self_attn_block: MultiHeadAttention, 
        cross_attn_block: MultiHeadAttention, 
        ff_block: FeedForwardBlock, 
        dropout: float
    ):
        super().__init__()
        self.self_attn_block = self_attn_block # masked self-attention (MHA) block
        self.cross_attn_block = cross_attn_block # cross attention (MHA) block taking input from encoder
        self.ff_block = ff_block # feedforward block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, enc_output, src_mask, trg_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attn_block(x, x, x, trg_mask))
        x = self.residual_connections[1](x, lambda x : self.cross_attn_block(x, enc_output, enc_output, src_mask))
        x = self.residual_connections[2](x, self.ff_block)
        return x # TODO: print out to see what this is outputting?
    
class Decoder(nn.Module):
    """Whole decoder"""
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, enc_output, src_mask, trg_mask):
        """Add & Norm"""
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, trg_mask) 
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """Projection layer"""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return self.proj(x)


class Transformer(nn.Module):
    """Main transformer architecture"""
    def __init__(self, 
                 encoder: Encoder,
                 decoder: Decoder, 
                 src_embed: InputEmbedding, 
                 trg_embed: InputEmbedding, 
                 src_pos: PositionalEncoding, 
                 trg_pos: PositionalEncoding, 
                 proj_layer: ProjectionLayer
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.proj_layer = proj_layer
            
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, enc_output, src_mask, trg, trg_mask):
        trg = self.src_embed(trg)
        trg = self.src_pos(trg)
        return self.decoder(trg, enc_output, src_mask, trg_mask)

    def project(self, x):
        return self.proj_layer(x)

def build_transformer(
        src_vocab_size: int, 
        trg_vocab_size: int, 
        src_seq_len: int, 
        trg_seq_len: int, 
        d_model=512,
        N=6, 
        H=8, 
        dropout=0.1, 
        d_ff=2048
    ):

    # embedding 
    src_embed = InputEmbedding(d_model, src_vocab_size)
    trg_embed = InputEmbedding(d_model, trg_vocab_size)

    # positional encodings  
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)


    # TODO: put encoder and decoder blocks initialization into `Encoder` and `Decoder`?  
    # encoder
    enc_blocks = []
    for _ in range(N):
        enc_self_attn_block = MultiHeadAttention(d_model, H, dropout) # mha == self_attn
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        enc_block =  EncoderBlock(d_model, enc_self_attn_block, ff_block, dropout)
        enc_blocks.append(enc_block)
    encoder = Encoder(d_model, nn.ModuleList(enc_blocks))

    # decoder
    dec_blocks = []
    for _ in range(N):
        dec_self_attn_block = MultiHeadAttention(d_model, H, dropout) # mha == self_attn
        dec_cross_attn_block = MultiHeadAttention(d_model, H, dropout) # mha == self_attn
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        dec_block =  DecoderBlock(d_model, dec_self_attn_block, dec_cross_attn_block, ff_block, dropout)
        dec_blocks.append(dec_block)
    decoder = Decoder(d_model, nn.ModuleList(dec_blocks))
    
    # projection layer 
    proj_layer = ProjectionLayer(d_model, trg_vocab_size)

    # main transformer arch. initialization
    tf = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, proj_layer)

    # init parameters with xavier initialization
    # TODO: add more options later? 
    for p in tf.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) 
    
    return tf

