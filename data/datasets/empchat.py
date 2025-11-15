# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Empathetic Dialogue Dataset Module

This module provides dataset classes for empathetic dialogue tasks including:
- EmpDataset: Main dataset class for training/evaluation
- RankingDataset: Dataset for knowledge ranking tasks

Classes handle loading dialogue data, generating knowledge from COMET models,
and preparing data for various downstream tasks.
"""
import heapq
import collections
import os
import pandas as pd
import random
import math
from collections import defaultdict
import yaml
from time import time
from typing import List, Dict, Tuple, Optional, Any
import sys
print(sys.path)
print(os.getcwd())
import contractions
import torch
from nltk import word_tokenize
from constant import EMOTION2LABEL, LABLE2EMOTION
from torch.utils.data import Dataset
from tqdm import trange
from transformers import BertTokenizer, AutoConfig,AutoTokenizer,AutoModel
import json
from question_seperate.model import BertForRanking
from data.util import get_score, get_sentence_similarity_model


# Configuration Constants
DEFAULT_HISTORY_LENGTH = 10
DEFAULT_SAMPLE_RATIO = 2
ENTITY_RELATIONS = ['ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty']
COMET_RELATIONS = ['xReact', 'xNeed', 'xIntent', 'xEffect', 'xWant']

# Text preprocessing configuration
WORD_PAIRS = {
    "it's": "it is", "don't": "do not", "doesn't": "does not", 
    "didn't": "did not", "you'd": "you would", "you're": "you are", 
    "you'll": "you will", "i'm": "i am", "they're": "they are", 
    "that's": "that is", "what's": "what is", "couldn't": "could not", 
    "i've": "i have", "we've": "we have", "can't": "cannot",
    "i'd": "i would", "aren't": "are not", "isn't": "is not", 
    "wasn't": "was not", "weren't": "were not", "won't": "will not", 
    "there's": "there is", "there're": "there are"
}

def get_social_knowledge_score(model, tokenizer, context, knowledge) -> List[Tuple[float, str]]:
    model.eval()
    if isinstance(knowledge, str):
        knowledge = [knowledge]
    
    # Tokenize context with attention mask
    context_tokens = tokenizer(context, return_tensors='pt')
    context_input_ids = context_tokens['input_ids'].squeeze(0)
    context_attention_mask = context_tokens['attention_mask'].squeeze(0)
    
    scores = []
    # Tokenize knowledge with attention mask
    knowledge_tokens = tokenizer(knowledge, return_tensors='pt', padding=True)
    knowledge_input_ids = knowledge_tokens['input_ids']
    knowledge_attention_mask = knowledge_tokens['attention_mask']
    
    if knowledge_input_ids.shape[0] == 1:
        knowledge_input_ids = knowledge_input_ids.squeeze(0)
        knowledge_attention_mask = knowledge_attention_mask.squeeze(0)
    
    with torch.inference_mode():
        # Get context embedding with attention mask
        context_embedding = model(
            context_input_ids.unsqueeze(0).to(model.device),
            attention_mask=context_attention_mask.unsqueeze(0).to(model.device)
        )[0][...,0,:]
        
        # Get knowledge embedding with attention mask
        knowledge_embedding = model(
            knowledge_input_ids.to(model.device),
            attention_mask=knowledge_attention_mask.to(model.device)
        )[0][...,0,:]
        
        score = torch.cosine_similarity(context_embedding, knowledge_embedding, dim=-1)
        scores = score.squeeze(0).tolist()
        scores_and_knowledge = list(zip(scores, knowledge))
        scores_and_knowledge.sort(key=lambda x: x[0], reverse=True)
    
    return scores_and_knowledge

# Text processing functions
def clean_text(sentence: str) -> str:
    """Clean and normalize text by expanding contractions and converting to lowercase."""
    sentence = sentence.lower()
    for contraction, expansion in WORD_PAIRS.items():
        sentence = sentence.replace(contraction, expansion)
    return sentence


def expand_contractions(text: str) -> str:
    """Expand contractions in text using the contractions library."""
    return " ".join([contractions.fix(word) for word in text.split()])




# Knowledge loading functions
def load_comet_knowledge(filepath: str) -> Dict[str, List]:
    """Load COMET knowledge from pickle file."""
    import pickle
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data



# Statistics generation function
def get_top_ngrams(dataset,k: int = 50) -> Tuple[List[str], List[str]]:
    """Generate top-k sentences and n-grams from the dataset."""
    data = [res for res in dataset.data['response']]
    
    new_data = []
    ngram = []  # 1-gram for now
    
    import re
    import numpy as np
    
    for r in data:
        new_r = ' '.join(re.split(r",|\?|!|\.", r))
        words = word_tokenize(new_r)
        for i in range(0, len(words) - 2):
            ngram.append(' '.join(words[i:i + 1]))
        
        r = re.split(r",|\?|!|\.", r)
        r = [item for item in r if len(item) > 0]
        new_data.extend(r)
    
    counter = collections.Counter(new_data)
    counts = counter.values()
    print(f"Standard deviation: {np.std(list(counts)) / math.log10(len(counts))}")
    
    topk_sentence = counter.most_common(k)
    ngram_counter = collections.Counter(ngram)
    top_k_ngram = ngram_counter.most_common(k)
    
    return topk_sentence, top_k_ngram


class AtomicRelationProcessor:
    """Processes ATOMIC relation templates for knowledge verbalization.https://arxiv.org/pdf/2010.05953"""
    
    def __init__(self):
        self.social_relation_map = self._initialize_social_relation_map()
        self.entity_relation_map = self._initialize_entity_relation_map()
    
    def _initialize_social_relation_map(self) -> Dict[str, Tuple[str, str]]:
        """Initialize mapping of relation types to templates."""
        return {
            'xReact': 'As a result, I feels',
            'xNeed': 'As a result, I needed',
            'xIntent': 'bacause I wanted',
            'xEffect': 'As a result, I will',
            'xWant': 'As a result, I wanted'
        }
    
    def _initialize_entity_relation_map(self) -> Dict[str, str]:
        """Initialize mapping of entity relations to human-readable forms."""
        return {
            'ObjectUse': 'is used for',
            'AtLocation': 'is located or found at',
            'MadeUpOf': 'is made up of',
            'HasProperty': 'can be characterized by being'
        }
    
    def get_atomic_relation_template(self, knowledge, rel_type: str) -> List[str]:
        """Convert atomic relation knowledge to verbalized form."""
        if rel_type in self.social_relation_map:
            template = self.social_relation_map.get(rel_type)
        elif rel_type in self.entity_relation_map:
            template = self.entity_relation_map.get(rel_type)
        else:
            raise ValueError(f"Unknown relation type: {rel_type}")
        
        if isinstance(knowledge, list):
            verbalized_knowl = []
            for item in knowledge:
                if item.lower().strip() == 'none':
                    continue
                verbalized_knowl.append(f'{template.replace("PersonX", "I")} {item.strip()}')
        elif isinstance(knowledge, str):
            verbalized_knowl = f'{template.replace("PersonX", "I")} {knowledge.strip()}'
        else:
            raise ValueError(f"Unknown knowledge type: {type(knowledge)}")
        
        return verbalized_knowl


class RankingDataset(Dataset):
    """Dataset class for knowledge ranking tasks."""
    
    def __init__(self, config_path: str, splitname: str, history_len: int = DEFAULT_HISTORY_LENGTH):
        """
        Initialize ranking dataset.
        
        Args:
            config_path: Path to YAML configuration file
            splitname: One of 'train', 'test', 'valid'
            history_len: Maximum number of dialogue history turns to consider
        """
        if splitname not in ['train', 'test', 'valid']:
            raise ValueError("splitname must be one of: train, test, valid")
        
        # 正确的YAML读取方式
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        
        self.data_folder = self.cfg.get('data_folder', None)
        self.bert_model_path = self.cfg.get('ranking_model_path', None)
        
        self.device = self.cfg.get('device', 'cuda:0')
        
        data_file = os.path.join(self.data_folder, f"DCKS-{splitname}.csv")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        self.max_hist_len = history_len
        self.sample_radio = DEFAULT_SAMPLE_RATIO
        self.entity_relation = ENTITY_RELATIONS
        
        # Initialize data structures
        self.ids = []
        self.content = []
        self.positive_knowl = []
        self.negative_knowl = []
        self.positive_knowl_counter = collections.defaultdict(int)
        
        # Load knowledge and tokenizer
        entity_knowledge_file = os.path.join(self.data_folder,self.cfg.get('entity_knowledge_file_name',None)[splitname])
        entity_knowledge = load_comet_knowledge(entity_knowledge_file)
        
        # Initialize BERT tokenizer using configuration
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_path)
        # self.tokenizer.add_special_tokens({'additional_special_tokens': ['[CSK]']})
        self._load_and_process_data(splitname, entity_knowledge)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[CSK]']})
        print(f"Positive examples: {len(self.positive_knowl)}")
        print(f"Negative examples: {len(self.negative_knowl)}")
        print(f"Dataset length: {max(len(self.positive_knowl), len(self.negative_knowl))}")
    
    def _load_and_process_data(self, splitname: str, entity_knowledge: Dict) -> None:
        """Load and process dialogue data with entity knowledge."""
        # Load data using pandas DataFrame for consistency with EmpDataset
        data_file = os.path.join(self.data_folder, f"DCKS-{splitname}.csv")
        df = pd.read_csv(data_file)
        
        history = []
        example_id = 0
        
        for i in trange(1, len(df), desc=f"Loading {splitname} data"):
            if df.iloc[i-1]['conv_id'] == df.iloc[i]['conv_id']:  # Same conversation
                c_uttr = clean_text(df.iloc[i-1]['utterence'].strip())
                prevsent = expand_contractions(c_uttr.replace("_comma_", ","))
                history.append(prevsent)
                
                turn_id = int(df.iloc[i]['utterence_idx'])
                if (turn_id % 2) == 0:
                    example_id += 1
                    self.positive_knowl.append([])
                    self.negative_knowl.append([])
                    
                    # Prepare dialogue history
                    prev_str = self.tokenizer.sep_token.join(
                        history[-self.max_hist_len:]
                    )
                
                    r_uttr = df.iloc[i]['utterence'].strip()
                    response = expand_contractions(r_uttr.replace("_comma_", ","))
                    self.content.append(prev_str)
                    
                    # Process entity knowledge
                    self._process_entity_knowledge(
                        entity_knowledge, example_id, prev_str, response
                    )
                
                self.ids.append((df.iloc[i]['conv_id'], df.iloc[i]['utterence_idx']))
            else:
                history = []
    
    def _process_entity_knowledge(self, entity_knowledge: Dict, 
                                example_id: int, prev_str: str, response: str) -> None:
        """Process entity knowledge for the current example."""
        for relation in entity_knowledge:
            know_list = entity_knowledge[relation][str(example_id - 1)]
            for item in know_list:
                for knowledge in item['knowledge']:
                    if knowledge.lower() == 'none':
                        continue
                    knowledge_str = f"{item['entity']} {relation} {knowledge}"
                    if (item['entity'] in prev_str and knowledge in response) or (item['entity'] in response and knowledge in prev_str):
                        self.positive_knowl[-1].append(knowledge_str)
                        self.positive_knowl_counter[knowledge_str] = 3
                    else:
                        self.negative_knowl[-1].append(knowledge_str)
    
    def __len__(self) -> int:
        return min(len(self.positive_knowl), len(self.negative_knowl))
    
    def __getitem__(self, index: int) -> Optional[Dict[str, List[torch.Tensor]]]:
        """Get training example at the specified index."""
        context = self.content[index]
        positive_knowl = self.positive_knowl[index]
        negative_knowl = self.negative_knowl[index]
        
        if len(positive_knowl) == 0 or len(negative_knowl) == 0:
            return None
        
        if self.sample_radio < 1:
            raise ValueError("sample_radio must be >= 1")
        if type(self.sample_radio) != int:
            raise ValueError("sample_radio must be an integer")
        
        # Sample positive knowledge
        post = random.choice(positive_knowl)
        if self.positive_knowl_counter[post] > 0:
            self.positive_knowl_counter[post] -= 1
        else:
            post = random.choice(positive_knowl)
        
        # Generate positive and negative examples
        pt, nt = [], []
        
        for _ in range(self.sample_radio):
            pos_text = f"{context}{self.tokenizer.sep_token} [CSK] {post}"
            neg_text = f"{context}{self.tokenizer.sep_token} [CSK] {random.choice(negative_knowl)}"
            
            pt.append(self.tokenizer(pos_text, return_tensors='pt')['input_ids'].squeeze(0))
            nt.append(self.tokenizer(neg_text, return_tensors='pt')['input_ids'].squeeze(0))
        
        return {
            "p_input_ids": pt,
            "n_input_ids": nt
        }

    def getid(self, index):
        return self.ids[index]
        


class EmpDataset(Dataset):
    """Main dataset class for empathetic dialogue tasks."""
    
    def __init__(
        self,
        config_path: str,
        splitname: str,
        history_len: int = DEFAULT_HISTORY_LENGTH,
        use_emotion: bool = False,
        use_social: bool = True,
        use_entity: bool = True
    ):
        """
        Initialize empathetic dialogue dataset.
        
        Args:
            opt_model: Model type for tokenization
            splitname: One of 'train', 'test', 'valid'
            history_len: Maximum dialogue history length
            use_emotion: Whether to include emotion labels
            use_social: Whether to use social knowledge
            use_entity: Whether to use entity knowledge
        """
        if splitname not in ['train', 'test', 'valid']:
            raise ValueError("splitname must be one of: train, test, valid")
        
        # 正确的YAML读取方式
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.model_for_generate=cfg.get('bart_model_path',None)
        self.model_for_entity_knowledge=cfg.get('ranking_model_path',None)
        self.model_for_social_knowledge=cfg.get('mpnet_model_path',None)
        self.data_folder=cfg.get('data_folder',None)
        self.device=cfg.get('device','cuda:0')

        if use_social:
            social_knowledge_file_name=cfg.get('social_knowledge_file_name',None)[splitname]
            self.social_knowledge_file=os.path.join(self.data_folder,social_knowledge_file_name)
        if use_entity:
            entity_knowledge_file_name=cfg.get('entity_knowledge_file_name',None)[splitname]
            self.entity_knowledge_file=os.path.join(self.data_folder,entity_knowledge_file_name)


        
        self.max_hist_len = history_len
        self.use_social = use_social
        self.use_entity = use_entity
        
        # Load cached data if available
        cache_file = os.path.join(self.data_folder, f"DCKS-{splitname}_dataset.json")
        if os.path.exists(cache_file):
            self._load_cached_data(cache_file)
            return
        
        # Initialize processors and models
        self.relation_processor = AtomicRelationProcessor()
        self._initialize_knowledge_models(use_social)
        
        # Load and process data
        data_file = os.path.join(self.data_folder, f"DCKS-{splitname}.csv")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        df = pd.read_csv(data_file)

        if not os.path.exists(cache_file):
            self._save_cached_data(cache_file)
            return
        self._process_dataset(df, self.model_for_generate, splitname, return_ids)
        self._save_cached_data(cache_file)
        print(f"Dataset length: {len(self.data['emotion'])}")
    
    def _load_cached_data(self, cache_file: str) -> None:
        """Load preprocessed data from cache."""
        st = time()
        print("Loading data from cache...")
        self.data = json.load(open(cache_file, 'r', encoding='utf-8'))
        et = time()
        print(f"Cache loaded in {et - st:.2f} seconds")
        
    def _save_cached_data(self, cache_file: str) -> None:
        """Save preprocessed data to cache."""
        st = time()
        print("Saving data to cache...")
        json.dump(self.data, open(cache_file, 'w', encoding='utf-8'),ensure_ascii=False,indent=4)
        et = time()
        print(f"Cache saved in {et - st:.2f} seconds")
    def _initialize_knowledge_models(self, use_social: bool) -> None:
        """Initialize knowledge models and processors if needed."""
        self.generate_model_tokenizer = AutoTokenizer.from_pretrained(self.model_for_generate)
        if self.use_entity:
            self.entity_score_model = BertForRanking.from_pretrained(self.model_for_entity_knowledge).to(self.device)
            self.entity_score_model_tokenizer = AutoTokenizer.from_pretrained(self.model_for_entity_knowledge)
            self.entity_score_model_tokenizer.add_special_tokens({'additional_special_tokens': ['[CSK]']})
            self.CSK_id = self.entity_score_model_tokenizer.convert_tokens_to_ids('[CSK]')
            self.entity_score_model.eval()
        if use_social:
            # Note: Knowledge files will be dynamically loaded based on splitname
            self.comet_knowledge = None
            self.entity_knowledge = None
            
            # Initialize ranking model for entity knowledge

            
            # Initialize similarity model for COMET knowledge scoring
            self.social_score_model_tokenizer = AutoTokenizer.from_pretrained(self.model_for_social_knowledge)
            self.social_score_model = AutoModel.from_pretrained(self.model_for_social_knowledge).to(self.device)
    
    def _process_dataset(self, df: pd.DataFrame, opt_model: str, 
                        splitname: str, ) -> None:
        """Process the dataset and extract features."""
        # Load knowledge sources if needed
        if self.use_social:
            self.comet_knowledge = load_comet_knowledge(self.social_knowledge_file)
            if self.use_entity:
                self.entity_knowledge = load_comet_knowledge(self.entity_knowledge_file)
        
        # Initialize data structure
        self.data = defaultdict(list)
        history = []
        example_id, turn_id = 0, 0
        
        for i in trange(1, len(df), desc=f"Loading {splitname} data"):
            if df.iloc[i-1]['conv_id'] == df.iloc[i]['conv_id']:
                c_uttr = clean_text(df.iloc[i-1]['utterence'].strip())
                c_uttr = expand_contractions(c_uttr.replace("_comma_", ","))
                r_uttr = df.iloc[i]['utterence'].strip()
                r_uttr = expand_contractions(r_uttr.replace("_comma_", ","))
                history.append(c_uttr)
                turn_id = int(df.iloc[i]['utterence_idx'])
                
                if (turn_id % 2) == 0:
                    example_id += 1
                    emotion = df.iloc[i]['emotion']
                    
                    # Prepare context
                    
                    context = self.generate_model_tokenizer.eos_token.join(history[-self.max_hist_len:])
                    response = r_uttr
                    self._process_knowledge(context, response, emotion, example_id)
                    self.data['context'].append(context)
                    self.data['response'].append(response)

            else:
                history = []
    

    def _process_knowledge(self, c_uttr: str, response: str, 
                                      emotion: str, example_id: int) -> None:
        """Process knowledge sources and prepare response data."""
        self.data['emotion'].append(EMOTION2LABEL[emotion])
        
        with torch.no_grad():
            if self.use_comet:
                self._process_social_knowledge(
                    c_uttr, example_id
                )
                
                if self.use_entity:
                    self._process_entity_knowledge(
                        c_uttr, example_id
                    )
    
    def _process_social_knowledge(self, context: str, example_id: int) -> None:
        """Process COMET knowledge for the current example."""
        for relation in self.comet_knowledge:
            know = self.comet_knowledge[relation][str(example_id - 1)] # because example_id is 1-indexed
            know = self.relation_processor.get_atomic_relation_template(know, relation)
            
            # Get knowledge scores and select best
            know_score_and_knowledge = get_social_knowledge_score(
                self.social_score_model, self.social_score_model_tokenizer, context, know
            )
            self.data[relation].append(know_score_and_knowledge[0][1]) # 按照分数排序的knowledge
            # self.data[relation].append(
            #     tokenizer(know[know_indices[0]], return_tensors='pt')['input_ids'].squeeze(0)
            # )
    
    def _process_entity_knowledge(self, c_uttr: str, example_id: int) -> None:
        """Process entity knowledge for the main dataset."""
        all_scores = []
        all_knowledge_entries = []
        
        # Collect all knowledge entries
        for relation, relation_data in self.entity_knowledge.items():
            know_list = relation_data[str(example_id - 1)]
            flatten_know_list = [
                {"entity": pair['entity'], "relation": relation, "knowledge": knowledge.strip()}
                for pair in know_list
                for knowledge in pair['knowledge']
                if knowledge.strip().lower() != 'none' and knowledge.strip().lower() != pair['entity'].strip().lower()
            ]
            all_knowledge_entries.extend(flatten_know_list)
        
        if not all_knowledge_entries:
            self.data['entity_knowledge'].append([])
            return
        
        # Process in batches to avoid OOM
        batch_size = 32  # Adjust based on your GPU memory
        all_scores = []
        
        for i in range(0, len(all_knowledge_entries), batch_size):
            batch_entries = all_knowledge_entries[i:i + batch_size]
            batch_input_strings = [
                f"{c_uttr}{self.entity_score_model_tokenizer.sep_token} [CSK] {entry['entity']} {entry['relation']} {entry['knowledge']}"
                for entry in batch_entries
            ]
            
            # Process batch
            inputs = self.entity_score_model_tokenizer(
                batch_input_strings, 
                padding=True, 
                return_tensors='pt'
            ).to(self.entity_score_model.device)
            
            with torch.no_grad():
                scores, _ = self.entity_score_model(inputs['input_ids'], inputs['attention_mask'])
                scores = scores.squeeze(1)
                all_scores.extend(scores.tolist())
            
            # Clear GPU memory
            del inputs, scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Get top 3 knowledge based on scores
        scores_with_indices = [(score, i) for i, score in enumerate(all_scores)]
        top_3_scores_indices = heapq.nlargest(3, scores_with_indices, key=lambda x: x[0])
        top_3_indices = [idx for _, idx in top_3_scores_indices]
        
        selected_know_list = [all_knowledge_entries[idx] for idx in top_3_indices]
        self.data['entity_knowledge'].append([
            f"{entry['entity']} {self.relation_processor.entity_relation_map[entry['relation']]} {entry['knowledge']}"
            for entry in selected_know_list
        ])



    def __len__(self) -> int:
        return len(self.data['response'])
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get dataset item at the specified index."""
        t=dict([(k, self.data[k][index]) for k in self.data])
        if not self.use_social:
            for k in COMET_RELATIONS:
                t.pop(k)
        if not self.use_entity:
            t.pop('entity_knowledge')
            for k in ENTITY_RELATIONS:
                t.pop(k)
        return t
    
    def getid(self, index: int) -> Tuple[str, str]:
        """Get conversation and utterance ID for the specified index."""
        return self.ids[index] if hasattr(self, 'ids') else ("", "")




if __name__ == '__main__':
    # Example usage
    config_path = '/home/huangfu/empdialogue_code/empatheticDialogue1/config.yaml'  # 配置文件路径
    
    # 测试EmpDataset
    dataset = EmpDataset(config_path, 'train', history_len=10, return_ids=True, use_comet=True, use_entity=True)
    t=dataset[0]
    print(t)
    print(f"EmpDataset loaded with {len(dataset)} examples")
    
    # 测试RankingDataset  
    # ranking_dataset = RankingDataset(config_path, 'test', history_len=10)
    # print(f"RankingDataset loaded with {len(ranking_dataset)} examples")
