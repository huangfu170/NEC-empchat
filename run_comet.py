"""
COMET Knowledge Generation Module

This module provides tools for generating commonsense knowledge using COMET models.
It includes utilities for entity extraction, knowledge generation, and processing
dialogue datasets with COMET-generated insights.
"""

import logging
import os
import pickle
import time
from collections import defaultdict, namedtuple
from functools import wraps
from typing import List, Dict, Any, Tuple
import yaml
import nltk
import torch
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import CFG
from data.datasets import empchat

# Configuration Constants
COMET_ENTITY_RELATIONS = ['ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty']
COMET_SOCIAL_RELATIONS = ['xReact', 'xNeed', 'xIntent', 'xEffect', 'xWant']
REMOVE_TOKEN_LIST = ['<s>', '</s>']
ENTITY_POS_TAGS = ['NNS', 'NN', 'NNP', 'NNPS']
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_BEAMS = 5
DEFAULT_NUM_SAMPLES = 5

# Data structures
EntityPromptDetail = namedtuple("EntityPromptDetail", ['entity', 'input_prompt', 'example_index'])

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("function_timing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def log_execution_time(func):
    """Decorator to measure and log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        logger.info(f"Function '{func.__name__}' completed. Duration: {execution_time:.6f} seconds")
        return result
    
    return wrapper


def replace_pronouns(text: str) -> str:
    """Replace first-person pronouns with third-person for COMET processing."""
    word_pairs = {
        ' I ': ' PersonX ',
        'we': 'they',
        'my': 'his',
        'myself': 'himself',
        'our': 'their',
        'me': 'PersonX',
        'my': "PersonX's"
    }
    
    for original, replacement in word_pairs.items():
        text = text.replace(original, replacement)
    return text


def clean_text_for_comet(text: str) -> str:
    """Clean text by removing special tokens for COMET processing."""
    for token in REMOVE_TOKEN_LIST:
        text = text.replace(token, '')
    return text.strip()


def is_valid_entity(word: str) -> bool:
    """Check if a word is a valid entity candidate."""
    if len(word) < 3:
        return False
    
    if not word.isalpha():
        return False
    
    return True


def extract_entities(text: str) -> List[str]:
    """Extract entities from input text using POS tagging."""
    stopwords_set = set(stopwords.words('english'))
    
    # Normalize punctuation
    text = text.replace('.', ' ').replace(',', ' ').replace('?', ' ')
    
    # Tokenize and tag
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    
    # Filter entities
    entities = [
        word for word, pos in tagged 
        if (pos in ENTITY_POS_TAGS and 
            word.lower() not in stopwords_set and 
            is_valid_entity(word))
    ]
    
    return list(set(entities))  # Remove duplicates


def chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def trim_batch(input_ids: torch.Tensor, pad_token_id: int, 
              attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
    """Remove padding columns from batch tensors."""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], 
               attention_mask[:, keep_column_mask])


def use_task_specific_params(model, task: str) -> None:
    """Update model config with task-specific parameters."""
    task_specific_params = model.config.task_specific_params
    
    if task_specific_params is not None:
        params = task_specific_params.get(task, {})
        model.config.update(params)


def load_pickle_data(filepath: str) -> Any:
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle_data(data: Any, filepath: str) -> None:
    """Save data to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


class Comet:
    """COMET model wrapper for knowledge generation."""
    
    def __init__(self, model_path, device):
        print(torch.cuda.device_count())
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.batch_size = 1

        # init params
        use_task_specific_params(self.model, "summarization")
        self.model.zero_grad()

    # @log_execution_time
    def generate(self, input_event, rel, batch_size=32):
        """Generate knowledge using COMET model."""
        if isinstance(input_event, list) and isinstance(rel, list):
            query = ["{} {} [GEN]".format(ie, r) for ie, r in zip(input_event, rel)]
        else:
            query = "{} {} [GEN]".format(input_event, rel)

        with torch.no_grad():
            query = self.tokenizer(
                query, return_tensors="pt", truncation=True, padding="max_length"
            ).to(self.device)
            input_ids, attention_mask = trim_batch(
                **query, pad_token_id=self.tokenizer.pad_token_id
            )
            all_decs = []
            for i in range(0, input_ids.size(0), batch_size):
                input_ids_batch = input_ids[i:i + batch_size, ...]
                attention_mask_batch = attention_mask[i:i + batch_size, ...]
                summaries = self.model.generate(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    decoder_start_token_id=None,
                    num_beams=5,
                    num_return_sequences=5,
                )
                dec = self.tokenizer.batch_decode(
                    summaries,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                all_decs.append(dec)

            return [item for sublist in all_decs for item in sublist]


@log_execution_time
def generate_social_knowledge(dataset, comet_model: Comet, output_filename: str) -> None:
    """Generate social commonsense knowledge for dataset examples."""
    knowledge_results = {}
    
    for relation in COMET_SOCIAL_RELATIONS:
        logger.info(f"Generating knowledge for relation: {relation}")
        knowledge_results[relation] = {}
        for example_idx,item in tqdm(enumerate(dataset), desc=f"Processing {relation}",total=len(dataset)):
            input_event = item['last_utterance']
            input_event = clean_text_for_comet(input_event)
            
            generated_knowledge = comet_model.generate(input_event, relation)
            knowledge_results[relation][str(example_idx)] = generated_knowledge
    
    save_pickle_data(knowledge_results, output_filename)
    logger.info(f"Social knowledge saved to: {output_filename}")


@log_execution_time
def generate_entity_knowledge(dataset, comet_model: Comet, output_filename: str) -> None:
    """Generate entity-based commonsense knowledge for dataset examples."""
    knowledge_results = {}
    
    for relation in COMET_ENTITY_RELATIONS:
        logger.info(f"Generating entity knowledge for relation: {relation}")
        knowledge_results[relation] = {}
        for example_idx, example in tqdm(enumerate(dataset), desc=f"Processing {relation}",total=len(dataset)):
            
            # Extract and process entities
            knowledge_results[relation][str(example_idx)] = []
            input_event = example['context']
            input_event = clean_text_for_comet(input_event)
            entities = extract_entities(input_event)
            
            if not entities:
                continue
            
            # Prepare prompts for knowledge generation
            prompts_to_generate = []
            for entity in entities:
                prompt = f"{input_event}, in this case, {entity}"
                prompts_to_generate.append(
                    EntityPromptDetail(entity, prompt, example_idx)
                )
            
            # Generate knowledge
            input_prompts = [item.input_prompt for item in prompts_to_generate]
            relations = [relation]*len(prompts_to_generate)
            
            knowledge_list = comet_model.generate(input_prompts, relations)
            
            # Process and store results
            for i, prompt_detail in enumerate(prompts_to_generate):
                start_idx = i * DEFAULT_NUM_SAMPLES
                end_idx = start_idx + DEFAULT_NUM_SAMPLES
                
                entity_knowledge = [
                    k.strip() for k in knowledge_list[start_idx:end_idx] 
                    if k != 'none'
                ]
                
                knowledge_results[relation][str(example_idx)].append(
                    {
                        'entity': prompt_detail.entity,
                        'knowledge': entity_knowledge
                    }
                )
            # if example_idx > 10:
            #     break
    
    save_pickle_data(knowledge_results, output_filename)
    logger.info(f"Entity knowledge saved to: {output_filename}")


def main():
    """Main execution function."""
    # Initialize configuration and model
    with open("/home/huangfu/empdialogue_code/empatheticDialogue1/config.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    logger.info("Initializing COMET model...")
    comet_model = Comet(cfg["comet_model_path"], cfg["device"])
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = empchat.EmpDataset(
        "/home/huangfu/empdialogue_code/empatheticDialogue1/config.yaml", 'train', 
        history_len=10, 
        return_ids=False, 
        use_comet=False, 
        use_emotion=False,
        use_entity=False
    )
    
    # Generate knowledge
    logger.info("Starting knowledge generation...")
    # generate_entity_knowledge(
    #     dataset, 
    #     comet_model,
    #     os.path.join(cfg['data_folder'], 'DCKS-all_test_comet_entity_pickle.pkl')
    # )
    generate_social_knowledge(
        dataset, 
        comet_model,
        os.path.join(cfg['data_folder'], 'DCKS-all_train_comet_social_pickle.pkl')
    )
    # t=load_pickle_data(os.path.join(cfg['data_folder'], 'DCKS-all_train_comet_entity_pickle.pkl'))
    # print(t)
    logger.info("Knowledge generation completed successfully!")


if __name__ == '__main__':
    main()
