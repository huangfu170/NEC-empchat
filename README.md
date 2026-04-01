# Non-Emotion-Centric Empathetic Dialogue Generation

# 非情感中心的共情对话生成

This is the official repository for the COLING 2025 paper: **"Non-Emotion-Centric Empathetic Dialogue Generation"**.

本仓库是 COLING 2025 论文 **"Non-Emotion-Centric Empathetic Dialogue Generation"** 的官方代码。

## Project Structure / 项目结构

```
├── main.py                  # Training entrypoint (supports DDP multi-GPU) / 训练入口（支持DDP多卡）
├── train.py                 # Training & validation logic / 训练和验证逻辑
├── run_comet.py             # COMET commonsense knowledge generation / COMET常识知识生成
├── train_entity_ranker.py   # Entity ranking model training / 实体排序模型训练
├── constant.py              # Emotion labels & relation mappings / 情感标签和关系映射
├── config.yaml              # Configuration file / 配置文件
├── model/
│   ├── model.py             # CustomBartForConditionalGeneration & BertForRanking / 模型定义
│   └── Decoder.py           # Custom BART decoder with knowledge attention / 带知识注意力的自定义解码器
├── data/
│   ├── util.py              # Beam search, metrics, utilities / Beam search、评估指标、工具函数
│   └── datasets/
│       ├── empchat.py       # EmpDataset & RankingDataset / 数据集类
│       └── loader.py        # DataLoader collate functions / 数据加载器
└── metrics/
    └── distinct/
        └── distinct.py      # Distinct-N metric / Distinct-N 指标
```

## Environment Setup / 环境配置

```bash
# Create virtual environment / 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# Install dependencies / 安装依赖
pip install -r requirements.txt

# Download NLTK data / 下载NLTK数据
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')"
```

## Pre-trained Models / 预训练模型

Download the following models and place them under `pretrained/`:

下载以下模型并放置到 `pretrained/` 目录下：

| Model / 模型 | Source / 来源 | Path / 路径 |
|---|---|---|
| BART-base | [facebook/bart-base](https://huggingface.co/facebook/bart-base) | `pretrained/bart-base/` |
| MPNet-base | [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | `pretrained/mpnet-base/` |
| BERT-base | [bert-base-uncased](https://huggingface.co/bert-base-uncased) | `pretrained/bert-base/` |
| COMET-ATOMIC 2020 | [COMET-ATOMIC_2020_BART](https://github.com/allenai/comet-atomic-2020) | `pretrained/comet-atomic_2020_BART/` |

## Data Preparation / 数据准备

### Quick Start (Recommended) / 快速开始（推荐）

You can skip Steps 1–4 entirely by downloading the pre-built dataset, which includes raw data, COMET knowledge, the trained entity ranking model, and dataset caches.

你可以通过下载预先构建好的数据集来跳过下面的 Step 1–4，该数据集已包含原始数据、COMET 知识、训练好的实体排序模型和数据集缓存。

Download link / 下载链接: [TODO: add link]

After downloading, place the files as follows / 下载后按如下结构放置：

```
├── DCKS-dataset/
│   ├── train.pkl                                # Raw data / 原始数据
│   ├── val.pkl
│   ├── test.pkl
│   ├── DCKS-all_train_comet_social_pickle.pkl   # COMET social knowledge / COMET社交知识
│   ├── DCKS-all_test_comet_social_pickle.pkl
│   ├── DCKS-all_train_comet_entity_pickle.pkl   # COMET entity knowledge / COMET实体知识
│   ├── DCKS-all_test_comet_entity_pickle.pkl
│   ├── DCKS-train_dataset.json                  # Dataset cache / 数据集缓存
│   └── DCKS-test_dataset.json
└── DCKS-entity_ranking_model_context_best/      # Trained entity ranking model / 训练好的实体排序模型
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── vocab.txt
```

Once placed, you can go directly to [Training / 训练](#training--训练).

放置完成后可直接跳到 [训练](#training--训练) 部分。

---

If you prefer to build the dataset from scratch, follow Steps 1–4 below.

如果你希望从头构建数据集，请按以下 Step 1–4 操作。

### Step 1: Prepare raw data / 准备原始数据

Place the EmpatheticDialogues dataset pickle files (`train.pkl`, `val.pkl`, `test.pkl`) under `DCKS-dataset/`.

将 EmpatheticDialogues 数据集的 pickle 文件（`train.pkl`、`val.pkl`、`test.pkl`）放到 `DCKS-dataset/` 目录下。

### Step 2: Generate COMET knowledge / 生成COMET知识

```bash
python run_comet.py --config_path config.yaml
```

This generates social and entity commonsense knowledge for each dialogue example.

为每条对话样本生成社交和实体常识知识。

### Step 3: Train entity ranking model / 训练实体排序模型

```bash
python train_entity_ranker.py --cuda-id 0 --config-path config.yaml
```

### Step 4: Build dataset cache / 构建数据集缓存

The first run of training will automatically build and cache the processed dataset (JSON files under `DCKS-dataset/`).

首次训练时会自动构建并缓存处理后的数据集（JSON 文件保存在 `DCKS-dataset/` 下）。

## Training / 训练

### Single GPU / 单卡训练

Set `gpu_ids: [0]` in `config.yaml`, then:

在 `config.yaml` 中设置 `gpu_ids: [0]`，然后：

```bash
python main.py --config_path config.yaml
```

### Multi-GPU DDP / 多卡DDP训练

Set `gpu_ids` to the list of GPU IDs you want to use:

将 `gpu_ids` 设置为要使用的 GPU ID 列表：

```yaml
runtime:
  gpu_ids: [0, 1, 2]
```

```bash
python main.py --config_path config.yaml
```

The training script uses `torch.multiprocessing.spawn` internally — no need for `torchrun`.

训练脚本内部使用 `torch.multiprocessing.spawn`，无需使用 `torchrun`。

## Configuration Reference / 配置参考

### General Settings / 通用设置

| Key | Description / 说明 | Default |
|---|---|---|
| `data_folder` | Dataset directory / 数据集目录 | `DCKS-dataset` |
| `model_save_folder` | Model save directory / 模型保存目录 | `model_save` |
| `bart_model_path` | BART model path / BART模型路径 | `pretrained/bart-base` |
| `mpnet_model_path` | MPNet model path / MPNet模型路径 | `pretrained/mpnet-base` |
| `device` | Default device / 默认设备 | `cuda:0` |

### Training Settings / 训练设置

| Key | Description / 说明 | Default |
|---|---|---|
| `gpu_ids` | GPU IDs for DDP / DDP使用的GPU列表 | `[0]` |
| `epoch` | Number of epochs / 训练轮数 | `10` |
| `per_gpu_batch_size` | Batch size per GPU / 每张卡的batch size | `4` |
| `wo_entity` | Disable entity knowledge / 关闭实体知识 | `false` |
| `wo_social` | Disable social knowledge / 关闭社交知识 | `false` |

### Contrastive Learning Settings / 对比学习设置

| Key | Description / 说明 | Default | Recommended Range / 推荐范围 |
|---|---|---|---|
| `CL` | Enable contrastive learning / 启用对比学习 | `true` | `true/false` |
| `CL_sample_num` | Negative samples per type / 每种负样本数量 | `3` | 2, 3, 5 |
| `alpha_for_CL` | Weight for semantic similarity vs LM score / 语义相似度与LM分数的权重 | `0.7` | 0.3, 0.5, 0.7, 0.9 |
| `self_generated` | Use model-generated negatives / 使用模型自生成负样本 | `true` | `true/false` |
| `emotion_nega` | Use emotion-based negatives / 使用情感负样本 | `true` | `true/false` |
| `high_freq_nega` | Use high-frequency negatives / 使用高频句负样本 | `true` | `true/false` |
| `train_beam_size_for_CL` | Beam size for self-generated negatives / 自生成负样本的beam size | `10` | 5, 10, 15 |
| `cl_max_candidates` | Max CL candidates per sample / 每个样本最大CL候选数 | `64` | 32, 64, 128 |
| `cl_ranking_margin` | Base margin for ranking loss / 排序损失基础margin | `0.01` | 0.005, 0.01, 0.05 |
| `cl_gold_bleu_threshold` | BLEU threshold to mask gold / 过滤金标准的BLEU阈值 | `0.99` | 0.95, 0.99 |
| `cl_bleu_ngram` | N-gram order for CL BLEU / CL中BLEU的n-gram阶数 | `2` | 2, 3 |

### CL Hyperparameter Tuning Guide / CL超参调优指南

The contrastive learning module supports three types of negative samples:

对比学习模块支持三种负样本：

1. **Batch-internal negatives** (always on when CL=true): Other responses in the same batch serve as negatives. / **Batch内负样本**（CL=true时始终开启）：同一batch中的其他回复作为负样本。

2. **Emotion-based negatives** (`emotion_nega`): Responses from different emotion categories. Controlled by `CL_sample_num`. / **情感负样本**（`emotion_nega`）：来自不同情感类别的回复，数量由 `CL_sample_num` 控制。

3. **Self-generated negatives** (`self_generated`): Model's own beam search outputs. Most expensive but most effective. Beam size controlled by `train_beam_size_for_CL`. / **自生成负样本**（`self_generated`）：模型自身beam search的输出。开销最大但效果最好，beam size 由 `train_beam_size_for_CL` 控制。

4. **High-frequency negatives** (`high_freq_nega`): Most common sub-sentences from training data. Low overhead. / **高频句负样本**（`high_freq_nega`）：训练数据中最常见的子句。开销很低。

**Tips / 建议:**
- Start with `emotion_nega` only for fast iteration, then add `self_generated` for best results. / 先只开 `emotion_nega` 快速迭代，再加 `self_generated` 获得最佳效果。
- `self_generated` significantly increases training time (~3x). Use multi-GPU DDP to compensate. / `self_generated` 会显著增加训练时间（约3倍），建议使用多卡DDP加速。
- `cl_max_candidates` > 64 may cause OOM on 24GB GPUs with batch_size=4. / `cl_max_candidates` > 64 在24GB显存、batch_size=4时可能OOM。

## Evaluation Metrics / 评估指标

The model is evaluated on:
- **BLEU-1/2/3/4**: N-gram overlap with reference responses / 与参考回复的N-gram重叠度
- **Distinct-1/2/3**: Generation diversity / 生成多样性
- **Perplexity**: Language model quality / 语言模型质量

Validation runs automatically after each epoch (starting from epoch 2). The best model (by BLEU-4) is saved to `model_save/best_model/`.

验证在每个 epoch 后自动运行（从第2个epoch开始）。最佳模型（按BLEU-4）保存到 `model_save/best_model/`。

## Citation / 引用

```bibtex
@inproceedings{huang2025non,
  title={Non-Emotion-Centric Empathetic Dialogue Generation},
  author={Huang, Yuanxiang},
  booktitle={Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025)},
  year={2025}
}
```

## License / 许可证

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

本项目采用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE)。
