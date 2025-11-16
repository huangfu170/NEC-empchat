# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding

def create_collate_fn(tokenizer):
    social_keys=['xReact', 'xNeed', 'xIntent', 'xEffect', 'xWant']
    def collate_fn(batch):
        tokenized_batch=defaultdict(list)
        prompt1='<s>The following knowledge facts are highly relevant to the left query:</s>'
        prompt2='<s>query:</s>'
        keys=batch[0].keys()
        if 'entity_knowledge' in keys:
            for i,item in enumerate(batch):
                if len(item['entity_knowledge'])>0:
                    batch[i]['context']=prompt2+item['context']+prompt1+tokenizer.eos_token.join(item['entity_knowledge'])
                else:
                    batch[i]['context']=prompt2+item['context']

        for k in keys:
            tokenized_batch[k] = [item[k] for item in batch if k in item]
        tokenized_batch.pop('entity_knowledge')
        new_tokenized_batch={}
        for k in tokenized_batch:
            if k not in ['emotion']:
                new_tokenized_batch[k]=tokenizer(tokenized_batch[k],return_tensors='pt',padding=True)
            else:
                new_tokenized_batch[k]=torch.tensor(tokenized_batch[k],dtype=torch.long)
        
        # Check if any social_keys are in new_tokenized_batch
        if any(key in new_tokenized_batch for key in social_keys):
            # Collect all present social_keys tensors
            present_social_keys = [key for key in social_keys if key in new_tokenized_batch]
            # Get the list of tensors to stack, pad to the same length
            social_tensors = [new_tokenized_batch[key]['input_ids'] for key in present_social_keys]
            # Find max length for padding
            max_len = max(t.size(1) for t in social_tensors)
            padded_social_tensors = [torch.nn.functional.pad(t, (0, max_len - t.size(1)), value=tokenizer.pad_token_id) for t in social_tensors]
            # Stack along new dimension (len(social_keys), batch, seq)
            stacked_social = torch.stack(padded_social_tensors, dim=1)  # (batch, num_social, seq)
            new_tokenized_batch['social_knowledge'] = stacked_social
            # Optionally, remove the individual keys
            for key in present_social_keys:
                new_tokenized_batch.pop(key)
        return new_tokenized_batch
    return collate_fn


