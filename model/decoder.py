import random
from typing import Optional, Tuple, List, Union

import math
import torch
from torch import nn
from transformers.models.bart.modeling_bart import BartAttention, BartConfig, BartPretrainedModel, \
    BartLearnedPositionalEmbedding, _make_causal_mask, _expand_mask, logger
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from data.util import EncoderOutputs, EncoderMask


class BartDecoderLayer(nn.Module):
    """
    Difference from original: adds Knowledge Cross-Attention + Gate Fusion after standard Cross-Attention.
    Original flow: Self-Attn -> Cross-Attn -> FFN
    This version: Self-Attn -> Cross-Attn -> Knowledge-Attn(gated) -> FFN
    / 与原版区别：在标准 Cross-Attention 之后新增了 Knowledge Cross-Attention + Gate Fusion。
    原版流程：Self-Attn → Cross-Attn → FFN
    本版流程：Self-Attn → Cross-Attn → Knowledge-Attn(gated) → FFN
    """
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # [New] Social knowledge cross-attention, same structure as encoder_attn but attends to knowledge encoding / [新增] 社交知识 cross-attention，结构与 encoder_attn 相同，但 attend 的是知识编码
        self.knowl_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # [New] LayerNorm for knowledge attention / [新增] 知识 attention 的 LayerNorm
        self.knowl_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        # [New] Gating linear layer: input is [knowl_attn_output; residual] concat, output is sigmoid weight / [新增] 门控线性层：输入为 [knowl_attn_output; residual] 拼接，输出 sigmoid 权重
        self.know_gate = nn.Linear(self.embed_dim * 2, 1)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # [Differs from original] encoder_hidden_states is an EncoderOutputs namedtuple that needs unpacking / [与原版不同] encoder_hidden_states 是 EncoderOutputs namedtuple，需要解包
        # Original receives a single tensor; here we split into dialogue encoding and social knowledge encoding / 原版直接接收单个 tensor；这里拆分为对话编码和社交知识编码
        src_hidden_states, knowl_hidden_states = encoder_hidden_states.x_encoded, encoder_hidden_states.social_knowledge_encoder_outputs
        src_hidden_attn_mask, knowl_attn_mask = encoder_attention_mask.x_mask, encoder_attention_mask.social_knowledge_mask
        residual = hidden_states

        # ========== Self Attention (same as original) / Self Attention（与原版相同）==========
        # past_key_value[0:2] are self-attn cached k/v
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # ========== Encoder Cross-Attention (same as original, but key/value come from unpacked dialogue encoding) / Encoder Cross-Attention（与原版相同，但 key/value 来自解包后的对话编码）==========
        cross_attn_present_key_value = None
        cross_attn_weights = None

        residual = hidden_states

        # past_key_value[2:4] are cross-attn cached k/v
        cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=src_hidden_states,
            past_key_value=cross_attn_past_key_value,
            attention_mask=src_hidden_attn_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        present_key_value = present_key_value + cross_attn_present_key_value

        # ========== [New] Knowledge Cross-Attention + Gate Fusion / [新增] Knowledge Cross-Attention + Gate Fusion ==========
        # Original does not have this block. Here we apply a third cross-attention over social knowledge encoding,
        # and use a sigmoid gate to control how much knowledge is fused in, avoiding knowledge noise in generation.
        # / 原版没有这一块。这里对社交知识编码做第三路 cross-attention，
        # 并通过 sigmoid 门控决定融入多少知识信息，避免知识噪声干扰生成。
        if knowl_hidden_states is not None:
            residual = hidden_states
            # [Differs from original] KV cache extended to [4:6], original only has [0:4] / [与原版不同] KV cache 扩展到 [4:6]，原版只有 [0:4]
            knowl_attn_past_key_value = past_key_value[4:6] if past_key_value is not None else None
            khidden_states, _, knowl_attn_present_key_value = self.knowl_attn(
                hidden_states=hidden_states,
                key_value_states=knowl_hidden_states,
                past_key_value=knowl_attn_past_key_value,
                attention_mask=knowl_attn_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                output_attentions=output_attentions,
            )
            khidden_states = nn.functional.dropout(khidden_states, p=self.dropout, training=self.training)
            # [New] Gate Fusion: weight = sigmoid([knowl_output; residual])
            # Instead of direct residual addition, lets the model adaptively control knowledge injection strength
            # / [新增] Gate Fusion：weight = sigmoid([knowl_output; residual])
            # 而非直接残差相加，让模型自适应地控制知识注入强度
            weight = self.know_gate(torch.cat([khidden_states, residual], dim=-1)).sigmoid()
            hidden_states = residual + weight * khidden_states
            hidden_states = self.knowl_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + knowl_attn_present_key_value

        # ========== FFN (same as original) / FFN（与原版相同）==========
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MyBartDecoder(BartPretrainedModel):
    """
    Differences from original BartDecoder:
    1. encoder_hidden_states is an EncoderOutputs namedtuple (dialogue encoding + social knowledge encoding), not a single tensor
    2. encoder_attention_mask is an EncoderMask namedtuple, carrying separate masks for the two streams
    3. Social knowledge encoding is a pooled CLS vector, so an all-ones mask (no padding) is generated
    4. Uses custom BartDecoderLayer (with Knowledge Cross-Attention)
    / 与原版 BartDecoder 的区别：
    1. encoder_hidden_states 是 EncoderOutputs namedtuple（含对话编码 + 社交知识编码），而非单个 tensor
    2. encoder_attention_mask 是 EncoderMask namedtuple，分别携带两路 mask
    3. 社交知识编码是 pooled 后的 CLS 向量，所以生成全 1 mask（无 padding）
    4. 使用自定义 BartDecoderLayer（含 Knowledge Cross-Attention）
    """
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        # [Differs from original] Uses custom BartDecoderLayer instead of the original / [与原版不同] 使用自定义 BartDecoderLayer 而非原版 BartDecoderLayer
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, inputs_embeds.device, past_key_values_length=past_key_values_length
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,  # [Differs from original] type is EncoderOutputs / [与原版不同] 类型为 EncoderOutputs
            encoder_attention_mask: Optional[torch.LongTensor] = None,  # [Differs from original] type is EncoderMask / [与原版不同] 类型为 EncoderMask
            head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # [Differs from original] Unpack dialogue encoding mask from EncoderOutputs, expand to 4D / [与原版不同] 从 EncoderOutputs 中解包对话编码的 mask，展开为 4D
        if encoder_hidden_states.x_encoded is not None and encoder_attention_mask.x_mask is not None:
            x_attention_mask = _expand_mask(encoder_attention_mask.x_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # [Differs from original] Generate all-ones mask for social knowledge encoding / [与原版不同] 为社交知识编码生成全 1 mask
        # Knowledge encoding is a CLS-pooled vector per entry with no padding, so mask is all ones / 因为知识编码是每条知识取 CLS pooling 后的向量，没有 padding，所以 mask 全为 1
        social_knowledge_mask = None
        if encoder_hidden_states.social_knowledge_encoder_outputs is not None:
            social_knowledge_outputs_mask = torch.ones(
                encoder_hidden_states.social_knowledge_encoder_outputs.size(0),
                encoder_hidden_states.social_knowledge_encoder_outputs.size(1),
                device=inputs_embeds.device
            )
            social_knowledge_mask = _expand_mask(social_knowledge_outputs_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        positions = self.embed_positions(input, past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        # [Differs from original] Repack expanded 4D masks into EncoderOutputs/EncoderMask for custom decoder layer / [与原版不同] 将展开后的 4D mask 重新包装为 EncoderOutputs/EncoderMask，传给自定义 decoder layer
        if encoder_hidden_states.social_knowledge_encoder_outputs is not None:
            encoder_hidden_states = EncoderOutputs(x_encoded=encoder_hidden_states.x_encoded, social_knowledge_encoder_outputs=encoder_hidden_states.social_knowledge_encoder_outputs)
            encoder_attention_mask = EncoderMask(x_mask=x_attention_mask, social_knowledge_mask=social_knowledge_mask)
        else:
            encoder_hidden_states = EncoderOutputs(x_encoded=encoder_hidden_states.x_encoded, social_knowledge_encoder_outputs=None)
            encoder_attention_mask = EncoderMask(x_mask=x_attention_mask, social_knowledge_mask=None)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
