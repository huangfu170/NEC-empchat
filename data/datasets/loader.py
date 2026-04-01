# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

SOCIAL_KEYS = ['xReact', 'xNeed', 'xIntent', 'xEffect', 'xWant']

ENTITY_KNOWLEDGE_PREFIX = '<s>The following knowledge facts are highly relevant to the left query:</s>'
QUERY_PREFIX = '<s>query:</s>'


def _inject_entity_knowledge(batch: List[Dict], tokenizer: PreTrainedTokenizer) -> None:
    """Prepend entity knowledge facts into each example's context in-place."""
    for item in batch:
        if item['entity_knowledge']:
            knowledge_str = tokenizer.eos_token.join(item['entity_knowledge'])
            item['context'] = QUERY_PREFIX + item['context'] + ENTITY_KNOWLEDGE_PREFIX + knowledge_str
        else:
            item['context'] = QUERY_PREFIX + item['context']


def _tokenize_fields(batch: List[Dict], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """Collect and tokenize all fields except entity_knowledge."""
    keys = [k for k in batch[0].keys() if k != 'entity_knowledge']
    result = {}
    for k in keys:
        values = [item[k] for item in batch]
        if k == 'emotion':
            result[k] = torch.tensor(values, dtype=torch.long)
        else:
            result[k] = tokenizer(values, return_tensors='pt', padding=True)
    return result


def _merge_social_knowledge(tokenized: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> None:
    """Stack individual social relation tensors into a single 'social_knowledge' tensor in-place."""
    present_keys = [k for k in SOCIAL_KEYS if k in tokenized]
    if not present_keys:
        return

    tensors = [tokenized[k]['input_ids'] for k in present_keys]
    max_len = max(t.size(1) for t in tensors)
    padded = [
        F.pad(t, (0, max_len - t.size(1)), value=tokenizer.pad_token_id)
        for t in tensors
    ]
    # shape: (batch, num_relations, seq_len)
    tokenized['social_knowledge'] = torch.stack(padded, dim=1)

    for k in present_keys:
        del tokenized[k]


def create_collate_fn(tokenizer: PreTrainedTokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        if 'entity_knowledge' in batch[0]:
            _inject_entity_knowledge(batch, tokenizer)

        tokenized = _tokenize_fields(batch, tokenizer)
        _merge_social_knowledge(tokenized, tokenizer)

        return tokenized

    return collate_fn
