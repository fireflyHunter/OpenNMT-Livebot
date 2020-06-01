"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch
from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.video_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, inputs, mask, imgs=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``
            imgs (FloatTensor): ``(batch_size, num_frames, model_dim)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """

        input_norm = self.layer_norm1(inputs)

        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask, attn_type='self')
        context = self.drop(context) + inputs
        context_norm = self.layer_norm2(context)

        out, _ = self.video_attn(imgs, imgs, context_norm, attn_type='self')
        out = self.drop(out) + context

        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class LivebotEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions):
        super(LivebotEncoder, self).__init__()

        self.embeddings = embeddings
        self.video_encoder = VideoEncoder(d_model, d_ff, heads, dropout, heads)

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, imgs=None, lengths=None):

        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        visual_out = self.video_encoder(imgs)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask, imgs=visual_out)
        out = self.layer_norm(out)

        return emb, (visual_out.transpose(0, 1).contiguous(), out.transpose(0, 1).contiguous()), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)


class VideoEncoder(nn.Module):

    def __init__(self, d_model, d_ff, n_head, dropout, n_block):
        super(VideoEncoder, self).__init__()
        self.layers = nn.ModuleList([VideoBlock(d_model, d_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)
class VideoBlock(nn.Module):

    def __init__(self, d_model, d_ff, n_head, dropout):
        super(VideoBlock, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, attn_type='self')[0])
        return self.feed_forward(out)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.drop(sublayer(self.norm(x)))

