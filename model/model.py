import os.path
from typing import Optional, Tuple, List, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel

from dataclasses import dataclass
from transformers.models.bart.modeling_bart import BartEncoder, BartConfig, BartPretrainedModel, logger

from config import CFG
from data.util import beam_search,EncoderOutputs,EncoderMask,Seq2SeqModelOutput,BeamSearchConfig
from model.Decoder import MyBartDecoder
from transformers.utils import ModelOutput




def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.size())
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def compute_lm_loss(logits, label, loss_func):
    shift_logits = logits.contiguous()
    shift_label = label.contiguous()
    return loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_label.view(-1))


def get_encoder():
    return BartModel.from_pretrained(os.path.join(cfg.data_prefix, 'bart-base/')).encoder


class BartModelCustom(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, args=None):
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = BartEncoder(config, self.shared)
        self.knowl_encoder = BartEncoder(config, self.shared) # social knowledge encoder
        # prompt for entity knowledge: <s>The following knowledge facts are highly relevant to the left query:</s>
        self.prompt = torch.tensor([0, 133, 511, 2655, 4905, 32, 2200, 4249, 7, 5, 314, 25860, 35, 2])
        self.decoder = MyBartDecoder(config, self.shared)
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_encoder_outputs(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        social_knowledge: torch.Tensor,
        social_knowledge_mask: torch.Tensor
    ):
        '''
        description:
        param {*} self
        param {torch.Tensor} x 输入encoder的文本input_ids(对话历史和回复)
        param {torch.Tensor} x_mask 输入encoder的文本对应的attention_mask
        param {torch.Tensor} social_knowledge 输入encoder的社会知识,形状为[bs,social_knowledge_num,social_knowledge_len]
        param {torch.Tensor} social_knowledge_mask 输入encoder的社会知识对应的attention_mask,形状为[bs,social_knowledge_num,social_knowledge_len]
        return {*} x_encoded, social_knowledge_encoder_outputs
        '''

        x_encoded = self.encoder(
            input_ids=x,
            attention_mask=x_mask,
        )[0]
        if social_knowledge is not None and social_knowledge_mask is not None:
            bs, social_knowledge_num, social_knowledge_len = social_knowledge.size()
            social_knowledge_encoder_outputs = [
                self.knowl_encoder(input_ids=social_knowledge[:, i, :], attention_mask=social_knowledge_mask[:, i, :])[0][:, 0, ...]
                for i in range(social_knowledge_num)
            ]
            social_knowledge_encoder_outputs = torch.stack(social_knowledge_encoder_outputs, dim=1) # batch size, social_knowledge_num, social_knowledge_len
        else:
            social_knowledge_encoder_outputs = None
        return EncoderOutputs(x_encoded=x_encoded, social_knowledge_encoder_outputs=social_knowledge_encoder_outputs)

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        social_knowledge: Optional[torch.LongTensor] = None,
        social_knowledge_mask: Optional[torch.Tensor] = None,
        emotion: Optional[torch.LongTensor] = None,
        emotion_nega: Optional[torch.LongTensor] = None,
        high_freq: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            # entity_encoder_outputs通常是None
            encoder_outputs = self.get_encoder_outputs(
                input_ids, attention_mask, social_knowledge, social_knowledge_mask
            )

        # emotion_pred = self.emo_linear(encoder_outputs.mean(dim=1) + knowl_encoder_outputs.mean(dim=1))
        # emotion_loss_func = CrossEntropyLoss()
        # emotion_loss = emotion_loss_func(emotion_pred, emotion)
        emotion_loss = 0
        encoder_mask = EncoderMask(x_mask=attention_mask, social_knowledge_mask=social_knowledge_mask)
        # TODO:对于无常识的部分应该直接给0

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=encoder_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs,
            encoder_hidden_states=encoder_outputs,
            encoder_attentions=encoder_outputs,
            emotion_loss=emotion_loss,
        )


class CustomBartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    authorized_missing_keys = [r"final_logits_bias", r"encoder\.version", r"decoder\.version"]

    def __init__(self, config: BartConfig, args=None):

        super().__init__(config)
        self.model = BartModelCustom(config, args)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.linear_layer = nn.Linear(config.d_model, config.d_model)
        self.pad_id = 1
        # self.ranking_model=BertForRanking(bert_config)
        self.is_training = True
        self.post_init()
        self.CL_flag = args.CL
        self.emotion_nega = args.emotion_nega
        self.high_freq_nega = args.high_freq_nega
        self.self_generated = args.self_generated

    def _load_knowl_encoder_weight(self):
        with torch.no_grad():
            encoder_dict = self.model.encoder.state_dict()
            knowl_encoder_dict = self.model.knowl_encoder.state_dict()
            for name in knowl_encoder_dict:
                knowl_encoder_dict[name].data.copy_(encoder_dict[name].data)
            self.model.knowl_encoder.load_state_dict(knowl_encoder_dict)

    def sample_from_model(self, input_ids=None, attention_mask=None, social_knowledge=None, social_knowledge_mask=None, emotion=None):

        beam_size = 5
        self.is_training = False
        batch_size = input_ids.size(0)
        cand_ids = beam_search(
            self, input_ids, attention_mask, social_knowledge, social_knowledge_mask, emotion,  beam_size, num_return_sequences=beam_size
        )["sequences"]
        self.is_training = True
        return cand_ids.view(batch_size, beam_size, -1)

    def ranking_loss(self, cos_distance, bleu_distance):
        margin = 0.01
        ones = torch.ones(cos_distance.size(), device=cos_distance.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        total_loss = loss_func(cos_distance, cos_distance, ones)

        n = cos_distance.size(1)
        for i in range(1, n):
            pos_score = cos_distance[:, :-i]
            neg_score = cos_distance[:, i:]
            same_mask = (torch.abs(bleu_distance[:, :-i] - bleu_distance[:, i:]) > margin).float()
            ones = torch.ones(pos_score.size(), device=cos_distance.device)
            loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='none')  # batch x i
            marginal_loss = loss_func(pos_score, neg_score, ones)
            if same_mask.sum() > 0:
                total_loss += (marginal_loss * same_mask).sum() / same_mask.sum()

        return total_loss

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        knowl: Optional[torch.LongTensor] = None,
        entity_knowledge: Optional[torch.LongTensor] = None,
        args=None,
    ):
        beam_size = args.inference_beam_size
        ret_dict = beam_search(
            self, input_ids, attention_mask, knowl, entity_knowledge, None, beam_size, num_return_sequences=beam_size
        )
        cand_ids = ret_dict["sequences"]
        cand_mask = (cand_ids != 1).long()
        if args.alpha_for_CL == 0:
            # 正常的beam search
            cand_ids = cand_ids.view(input_ids.size(0), beam_size, -1)
            return cand_ids[:, 0, :]

        cand_len = torch.sum(cand_mask, dim=-1)
        max_len = torch.max(cand_len).item()
        cand_ids = cand_ids[:, :max_len]
        beam_indices = ret_dict['beam_indices']
        beam_indices = torch.where(beam_indices > 0, beam_indices, 0)
        decoder_hidden_states = ret_dict['decoder_hidden_states']
        hidden_states_from_output = torch.cat(
            [decoder_hidden_states[i] for i in range(len(decoder_hidden_states))], dim=1
        )
        h = hidden_states_from_output.size(-1)
        decoder_hidden_states = torch.gather(
            hidden_states_from_output, 0, beam_indices[:, :-1][:, :, None].expand(-1, -1, h)
        )
        encoder_hidden_states = ret_dict["encoder_hidden_states"]
        encoder_feature = self.affine_transformation(encoder_hidden_states, attention_mask)  # batch x h
        decoder_feature = self.affine_transformation(decoder_hidden_states, cand_mask[:, :-1])
        decoder_feature = decoder_feature.view(input_ids.size(0), beam_size, -1)  # batch x sample_num x h
        cos_distance = torch.cosine_similarity(
            encoder_feature.unsqueeze(1), decoder_feature, dim=-1
        )  # batch x sample_num
        scores = ret_dict["sequences_scores"].view(input_ids.size(0), -1)  # 最后一个词对应的概率 (bs,)
        normalize = torch.sum(0 - scores, keepdim=True, dim=-1)
        score = (1 - args.alpha_for_CL) * (scores / normalize) + args.alpha_for_CL * cos_distance
        cand_ids = cand_ids.view(input_ids.size(0), beam_size, -1)
        max_indices = torch.argmax(score, dim=-1)[:, None, None]
        dummy = max_indices.repeat(1, 1, cand_ids.size(2))
        return torch.gather(cand_ids, 1, dummy).squeeze(1)

    def affine_transformation(self, input_features, padding_mask, axis=1):
        length = torch.sum(padding_mask, axis=1) - 1
        max_len = int(padding_mask.shape[-1]) if padding_mask.shape[-1] is not None else int(length.max())
        batch_size = length.shape[0]
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(length)
        padding_mask = broad_cast_seq_len < length.unsqueeze(1)
        trans_tmp = F.relu(self.linear_layer(input_features))  # batch
        trans_tmp = trans_tmp * padding_mask.unsqueeze(-1).float()
        trans_emb = torch.sum(trans_tmp, axis=axis)
        return trans_emb * (1 / length.unsqueeze(-1))

    def form_ngram(self, input_tensor, n=3):
        """
        input_tensor: batch x sample_num x seq_len
        return: batch x seq_len-3 x 4
        """
        bsz, cand_num, seq_len = input_tensor.size(0), input_tensor.size(1), input_tensor.size(2)
        seq_len_clip = seq_len - n + 1
        input_tensor_repeated = input_tensor[:, :, None, :].repeat(1, 1, seq_len_clip, 1)
        help_matrix_1 = torch.triu(torch.ones(seq_len, seq_len))
        help_matrix_2 = torch.triu(torch.ones(seq_len, seq_len), diagonal=n)
        help_matrix = (help_matrix_1 - help_matrix_2)[:seq_len_clip].bool()[None, None, :, :]
        ret_tensor = torch.masked_select(input_tensor_repeated, help_matrix.to(input_tensor.device))
        return ret_tensor.view(bsz, cand_num, seq_len_clip, n)

    def pad2max_len(self, input_tensor, max_len):
        pad_size = max_len - input_tensor.shape[-1]
        pad_tensor = torch.full(
            [input_tensor.shape[0], input_tensor.shape[1], pad_size], 1, device=input_tensor.device
        ).long()
        return torch.cat([input_tensor, pad_tensor], dim=-1)

    def torch_bleu(self, ref_tensor, sys_tensor, pad_id, n_gram=3):
        """
        Calculates n-gram precision with brevity penalty. contributed by jinulee-v

        ref_tensor: batch x seq_len1
        sys_tensor: batch x sample_num x seq_len2
        """
        # Determine batch size, sample count(=beam size), n-gram
        bsz, sample_num = sys_tensor.size(0), sys_tensor.size(1)
        n = min(min(n_gram, ref_tensor.size(-1)), sys_tensor.size(-1))

        # Generate masks
        ref_padding = (~(ref_tensor == pad_id)).float()
        ref_padding[:, 0] = 1
        ref_ngram_mask = torch.arange(0, ref_padding.size(1), device=ref_padding.device) * torch.ones_like(ref_padding)
        ref_ngram_mask = torch.where(
            ref_ngram_mask < (torch.sum(ref_padding, dim=-1, keepdims=True) - n + 1),
            ref_padding,
            torch.zeros_like(ref_padding),
        )[:, : ref_ngram_mask.size(-1) - n + 1]
        sys_padding = (~(sys_tensor == pad_id)).float()
        sys_padding[:, 0] = 1
        sys_ngram_mask = torch.arange(0, sys_padding.size(-1), device=sys_padding.device) * torch.ones_like(sys_padding)
        sys_ngram_mask = torch.where(
            sys_ngram_mask < (torch.sum(sys_padding, dim=-1, keepdims=True) - n + 1),
            sys_padding,
            torch.zeros_like(sys_padding),
        )[:, :, : sys_ngram_mask.size(-1) - n + 1]

        # Get n-grams
        ref_tensor = ref_tensor * ref_padding  # mask out paddings
        sys_tensor = sys_tensor * sys_padding
        ref_tensor = ref_tensor[:, None, :].repeat(1, sample_num, 1)  # readjust ref size to match sys
        input_tensor1_ngram = self.form_ngram(ref_tensor, n).float()
        # batch x sample_num x seq_len-(n-1) x n
        input_tensor2_ngram = self.form_ngram(sys_tensor, n).float()

        # Calculate similarity matrix
        sim_matrix = (
            torch.norm(  # Calculate L2 norm to find if N-gram in `sys`` is present in `ref``
                input_tensor2_ngram.unsqueeze(3) - input_tensor1_ngram.unsqueeze(2), p=2, dim=-1
            )
            == 0.0
        ).to(torch.float)
        # print(sim_matrix.size(), sys_ngram_mask.size(), ref_ngram_mask.size())
        sim_matrix *= sys_ngram_mask.unsqueeze(3) * ref_ngram_mask.unsqueeze(1).unsqueeze(2)
        sim_matrix = torch.sum(torch.max(sim_matrix, dim=-1).values, dim=-1)

        # Brevity penalty
        ref_len = torch.sum(ref_padding, dim=-1, keepdims=True)
        sys_len = torch.sum(sys_padding, dim=-1)
        bp = torch.exp(1 - (ref_len / sys_len))
        bp = torch.where(ref_len >= sys_len, bp, torch.ones_like(bp))

        # batch x sample_num
        return sim_matrix / torch.sum(sys_ngram_mask, dim=-1) * bp

    def unify_length(self, input_tensor, source_length, max_length):
        if source_length < max_length:
            # 长就截断，短就补齐
            input_tensor = self.pad2max_len(input_tensor, max_length)
        else:
            input_tensor = input_tensor[:, :, :max_length]
        return input_tensor

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        social_knowledge: Optional[torch.LongTensor] = None,
        social_knowledge_mask: Optional[torch.Tensor] = None,
        emotion: Optional[torch.LongTensor] = None,
        emotion_nega: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        high_freq: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_input_ids!=self.config.pad_token_id,
            social_knowledge=social_knowledge,
            social_knowledge_mask=social_knowledge_mask,
            emotion=emotion,
            emotion_nega=emotion_nega,
            high_freq=high_freq,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs.last_hidden_state)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        batch_size = lm_logits.size(0)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        cl_loss = torch.tensor([0]).to(lm_logits.device)

        if self.is_training and self.CL_flag:
            cand_len, emotion_cand_len, high_freq_cand_len = 0, 0, 0
            encoder_outputs= self.model.get_encoder_outputs(
                input_ids, attention_mask, social_knowledge, social_knowledge_mask
            )
            # ! 这里应该是个隐藏的bug，需要检查他们是否存在
            if self.self_generated: 
                cand_ids = self.sample_from_model(input_ids=input_ids, attention_mask=attention_mask, social_knowledge=social_knowledge, social_knowledge_mask=social_knowledge_mask, emotion=emotion)
                cand_len = cand_ids.size(2)
            if self.emotion_nega:
                emotion_nega=emotion_nega.to(lm_logits.device)
                emotion_cand_ids = emotion_nega.view(batch_size, 3, -1)
                emotion_cand_len = emotion_cand_ids.size(-1)
            if self.high_freq_nega:
                high_freq_cand_ids = high_freq.unsqueeze(0).repeat(batch_size, 1, 1)
                high_freq_cand_len = high_freq_cand_ids.size(-1)
            samples_from_batch = decoder_input_ids[None, :, :].repeat(batch_size, 1, 1)
            samples_len = samples_from_batch.size(2)
            #! 这里要注意是否是个tensor
            max_length = max(samples_len, high_freq_cand_len, emotion_cand_len, cand_len)
            samples_from_batch = self.unify_length(samples_from_batch, samples_len, max_length)
            samples_all = [samples_from_batch]
            if emotion_cand_ids is not None:
                emotion_cand_ids = self.unify_length(emotion_cand_ids, emotion_cand_len, max_length)
                samples_all.append(emotion_cand_ids)
            if high_freq_cand_ids is not None:
                high_freq_cand_ids = self.unify_length(high_freq_cand_ids, high_freq_cand_len, max_length)
                samples_all.append(high_freq_cand_ids)
            if cand_ids is not None:
                cand_ids = self.unify_length(cand_ids, cand_len, max_length)
                samples_all.append(cand_ids)

            samples_all = torch.cat(samples_all, dim=1)
            actual_distance = self.torch_bleu(decoder_input_ids, samples_all, self.pad_id, 2)

            distance_mask = actual_distance < 0.99  # use to mask the gold
            actual_distance_masked = actual_distance * distance_mask.float()

            sample_num = min(64, actual_distance_masked.size(1) - 1)
            actual_distance, actual_indices = torch.sort(actual_distance_masked, dim=-1, descending=True)
            # 对每个样本选出bleu值最大的sample_num个样本,扔掉最差的一个
            sampled_actual_distance = actual_distance[:, :sample_num]
            sampled_actual_indices = actual_indices[:, :sample_num]

            self_indices = torch.arange(0, batch_size).reshape(batch_size, 1).to(
                sampled_actual_indices.device
            ) + cand_ids.size(
                1
            )  # manually add gold

            sampled_indices = torch.cat([self_indices, sampled_actual_indices], dim=-1)

            self_distance = torch.full([batch_size, 1], 1.0, device=sampled_actual_distance.device)
            sampled_bleu_distance = torch.cat([self_distance, sampled_actual_distance], dim=-1)
            dummy = sampled_indices.unsqueeze(-1).repeat(1, 1, samples_all.size(2))
            # batch x sample_num x seq_len
            sampled_input = torch.gather(samples_all, 1, dummy)

            decoder_hidden_states = []
            for sample_idx in range(sampled_indices.size(-1)):
                sampled_input_dec = sampled_input[:, sample_idx, :]

                sample_pad_mask = ~(sampled_input_dec == self.pad_id)
                sample_pad_mask[:, 0] = 1

                decoder_out = self.model.decoder(
                    input_ids=sampled_input_dec,
                    attention_mask=sample_pad_mask,
                    encoder_hidden_states=encoder_outputs,
                    encoder_attention_mask=EncoderMask(x_mask=attention_mask),
                )

                decoder_feature = decoder_out.last_hidden_state  # batch x tgt_len x hidden
                decoder_feature = self.affine_transformation(decoder_feature, sample_pad_mask)  # batch x h
                decoder_hidden_states.append(decoder_feature.unsqueeze(1))

            encoder_feature = self.affine_transformation(encoder_outputs.x_encoded, attention_mask)  # batch x h
            decoder_feature = torch.cat(decoder_hidden_states, dim=1)  # batch x sample_num x h
            cos_distance = torch.cosine_similarity(
                encoder_feature.unsqueeze(1), decoder_feature, dim=-1
            )  # batch x samle_num
            cl_loss = self.ranking_loss(cos_distance, sampled_bleu_distance)
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return {
            "masked_lm_loss": masked_lm_loss,
            "cl_loss": cl_loss,
            "lm_logits": lm_logits,
            "decoder_hidden_states": outputs[0],
        }


class BertForRanking(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForRanking, self).__init__(config)
        # self.bert = DebertaV2Model(config)
        self.bert = BertModel(config)

        self.multi_stage_discriminator = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(768, 768 // 2), nn.Tanh(), nn.Linear(768 // 2, 1, bias=False), nn.Sigmoid()
        )
        self.cls_discriminator = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(768, 768 // 2), nn.ReLU(), nn.Linear(768 // 2, 2)
        )
        self.post_init()

    def forward(self, x, x_mask, labels=None):

        csk_position = (x == 1).nonzero() # unused token position 1
        x = self.bert(x, attention_mask=x_mask)[0]
        x = torch.stack([x[i, csk_position[i, 1], :] for i in range(x.size(0))], dim=0)

        x = F.normalize(x, dim=-1)

        margin_y = self.multi_stage_discriminator(x)
        cls_y = self.cls_discriminator(x)

        return margin_y, cls_y
