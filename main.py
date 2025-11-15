import argparse
import logging
from datetime import datetime
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, AutoConfig

from data.datasets.empchat import EmpDataset
from data.util import seed_everything
from data.datasets.loader import create_collate_fn
from question_seperate.model import CustomBartForConditionalGeneration
from train import train_validate
import yaml


def load_config():
    """
    Load YAML once so both runtime and static settings share one namespace.
    Only the config path itself stays on the CLI.
    """
    parser = argparse.ArgumentParser(
        description="Empathetic dialog training entrypoint.")
    parser.add_argument(
        "--config_path",
        default="config.yaml",
        help="Path to the unified configuration file."
    )
    cli_args = parser.parse_args()
    with open(cli_args.config_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    runtime_cfg = cfg_dict.pop('runtime', None)
    if runtime_cfg is None:
        raise KeyError(
            "Missing `runtime` section in config.yaml. "
            "Move the previous argparse defaults there."
        )
    cfg_dict.update(runtime_cfg)
    cfg_dict['config_path'] = cli_args.config_path
    return SimpleNamespace(**cfg_dict)


if __name__ == '__main__':
    cfg = load_config()

    if any([cfg.self_generated, cfg.emotion_nega, cfg.high_freq_nega]):
        logging_file_name = "CL"
        logging_file_name += f"_alpha{cfg.alpha_for_CL}"
        logging_file_name += f"_tbs{cfg.train_beam_size_for_CL}"
    else:
        logging_file_name = ""

    if not cfg.wo_entity:
        logging_file_name += '_ek'
    logging_file_name += f"_ibs{cfg.inference_beam_size}"
    logging_file_name += f"_bs{cfg.batch_size}"
    logging.basicConfig(
        level=logging.INFO,
        filename=datetime.now().strftime('%Y-%m-%d') + logging_file_name + '.log'
    )
    logging.info(f"{cfg}")

    seed_everything(42)

    rel_list = ['xReact', 'xNeed', 'xIntent', 'xEffect', 'xWant']
    entity_rel_list = ['ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty']
    tokenizer = BartTokenizer.from_pretrained(cfg.bart_model_path)
    knowledge_prompt = '<s>The following knowledge facts are highly relevant to the left query:</s>'
    bart_config = AutoConfig.from_pretrained(cfg.bart_model_path)
    device = torch.device(
        f'cuda:{cfg.cuda_id}' if torch.cuda.is_available() else 'cpu')
    model = CustomBartForConditionalGeneration.from_pretrained(
        cfg.bart_model_path,
        config=bart_config, args=cfg
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    history_len = getattr(cfg, 'history_length', 10)

    dataset = EmpDataset(
        config_path=cfg.config_path,
        splitname='train',
        history_len=history_len,
        use_social=not cfg.wo_social,
        use_entity=not cfg.wo_entity
    )
    dataloader = DataLoader(
        dataset,
        cfg.batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(tokenizer)
    )

    valid_dataset = EmpDataset(
        config_path=cfg.config_path,
        splitname='test',
        history_len=history_len,
        use_social=not cfg.wo_social,
        use_entity=not cfg.wo_entity
    )
    valid_dataloader = DataLoader(
        valid_dataset, 16, shuffle=False,
        collate_fn=create_collate_fn(tokenizer))
    train_validate(
        model,
        optimizer,
        dataloader,
        valid_dataloader,
        tokenizer,
        device,
        None,
        cfg,
        0,
        cfg.epoch
    )
