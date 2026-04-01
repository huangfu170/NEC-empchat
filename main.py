import argparse
import os
from types import SimpleNamespace

import swanlab

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BartTokenizer, AutoConfig

from data.datasets.empchat import EmpDataset
from data.util import seed_everything
from data.datasets.loader import create_collate_fn
from model.model import CustomBartForConditionalGeneration
from train import train_validate
import yaml


def load_config():
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


def setup_distributed(rank, world_size, gpu_ids):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    device_id = gpu_ids[rank]
    torch.cuda.set_device(device_id)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return torch.device(f'cuda:{device_id}')


def cleanup_distributed():
    dist.destroy_process_group()


def main(rank, world_size, cfg, gpu_ids):
    device = setup_distributed(rank, world_size, gpu_ids)

    if rank == 0:
        swanlab.init(project="NEC-empchat", config=vars(cfg))

    seed_everything(42 + rank)

    tokenizer = BartTokenizer.from_pretrained(cfg.bart_model_path)
    bart_config = AutoConfig.from_pretrained(cfg.bart_model_path)
    model = CustomBartForConditionalGeneration.from_pretrained(
        cfg.bart_model_path, config=bart_config, args=cfg
    ).to(device)
    model = DDP(model, device_ids=[gpu_ids[rank]], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    history_len = getattr(cfg, 'history_length', 10)

    dataset = EmpDataset(
        config_path=cfg.config_path,
        splitname='train',
        history_len=history_len,
        use_social=not cfg.wo_social,
        use_entity=not cfg.wo_entity
    )
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.per_gpu_batch_size,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=create_collate_fn(tokenizer)
    )

    valid_dataset = EmpDataset(
        config_path=cfg.config_path,
        splitname='test',
        history_len=history_len,
        use_social=not cfg.wo_social,
        use_entity=not cfg.wo_entity
    )
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    valid_dataloader = DataLoader(
        valid_dataset, 16, shuffle=False,
        sampler=valid_sampler,
        collate_fn=create_collate_fn(tokenizer)
    )

    train_validate(
        model, optimizer, dataloader, valid_dataloader,
        tokenizer, device, None, cfg, 0, cfg.epoch, rank, world_size
    )

    if rank == 0:
        swanlab.finish()
    cleanup_distributed()


if __name__ == '__main__':
    cfg = load_config()
    gpu_ids = cfg.gpu_ids
    world_size = len(gpu_ids)
    mp.spawn(main, args=(world_size, cfg, gpu_ids), nprocs=world_size, join=True)
