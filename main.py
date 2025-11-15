import argparse
import logging
import os.path
from datetime import datetime

import torch

from torch.utils.data import DataLoader
from transformers import BartTokenizer, AutoConfig

from config import CFG
from data.datasets.empchat import EmpDataset
from data.util import seed_everything
from data.datasets.loader import create_collate_fn
from question_seperate.model import CustomBartForConditionalGeneration
from train import get_top, train_validate
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda-id', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--wo_entity", action='store_true',
                        help="Use entity knowledge")
    # TODO: 关于社交常识的消融不能直接把这部分扔掉，不然中间的attn没有计算的办法（除非把attn也扔掉，但那就相当于纯BART）
    parser.add_argument("--wo_social", action='store_true',
                        help="Use social knowledge")
    parser.add_argument("--alpha_for_CL", default=0.7, help="the alpha for CL")
    parser.add_argument("--high_freq_nega", action='store_true', 
                        help="Use high frequency negative sampling")
    parser.add_argument("--inference_beam_size", default=5,
                        help="Beam size for inference")
    parser.add_argument("--train_beam_size_for_CL", default=10,
                        help="Beam size for CL learning in training stage")
    parser.add_argument("--self-generated",action='store_true',help="是否使用自己生成的作为负样例")
    parser.add_argument("--emotion_nega", action='store_true', help="是否使用情绪负样例")
    parser.add_argument("--config_path", default="config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()
    with open(args.config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)


    if any([args.self_generated, args.emotion_nega,args.high_freq_nega]):
        logging_file_name = "CL"
        logging_file_name += f"_alpha{args.alpha_for_CL}"

        logging_file_name += f"_tbs{args.train_beam_size_for_CL}"

    else:
        logging_file_name = ""


    if not args.wo_entity:
        logging_file_name += '_ek'
    logging_file_name += f"_ibs{args.inference_beam_size}"
    logging_file_name += f"_bs{args.batch_size}"
        # logging_file_name += f"_bs{args.batch_size}_emoshff_woself"
    logging.basicConfig(level=logging.INFO, filename=datetime.now().strftime(
        '%Y-%m-%d') + logging_file_name + '.log')
    logging.info(f"{args}")


    seed_everything(42)


    rel_list = ['xReact', 'xNeed', 'xIntent', 'xEffect', 'xWant']
    entity_rel_list = ['ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty']
    # 按照config.yaml里的配置项进行初始化
    tokenizer = BartTokenizer.from_pretrained(cfg['bart_model_path'])
    knowledge_prompt='<s>The following knowledge facts are highly relevant to the left query:</s>'
    # ans = tokenizer.decode(torch.tensor(
    #     [0, 133, 511, 2655, 4905, 32, 2200, 4249, 7, 5, 314, 25860, 35, 2]))
    bart_config = AutoConfig.from_pretrained(cfg['bart_model_path'])
    device = torch.device(
        f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    model = CustomBartForConditionalGeneration.from_pretrained(
        cfg['bart_model_path'],
        config=bart_config, args=args
    ).to(device)

    # model.load_state_dict(torch.load(os.path.join(
    #     cfg.model_prefix, "DCKS-CL_gen_model_tbs10_bs8(1).pt")), strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_scores = []

    dataset = EmpDataset(
        config_path=args.config_path,
        splitname='train', 
        history_len=10,
        use_social=not args.wo_social,
        use_entity=not args.wo_entity
    )
    dataloader = DataLoader(dataset, args.batch_size,
                            shuffle=False, collate_fn=create_collate_fn(tokenizer))

    valid_dataset = EmpDataset(
        config_path=args.config_path,
        splitname='test', 
        history_len=10,
        use_social=not args.wo_social,
        use_entity=not args.wo_entity
    )
    valid_dataloader = DataLoader(
        valid_dataset, 16, shuffle=False, collate_fn=create_collate_fn(tokenizer))
    train_validate(model, optimizer, dataloader, valid_dataloader,
                   tokenizer, device, None, args, 0, args.epoch)
    # res=validate(model, valid_dataloader, tokenizer, device, 0)
    # print(res)
    # t_dataset = empchat.EmpDataset('bart-base', 'test', '/home/yxhuangfu/empatheticDialogue1/DCKS-dataset',
    #                                    history_len=10,return_ids=False,use_comet=False)
    # generate_outputs(model,t_dataset, valid_dataloader, tokenizer, "/home/yxhuangfu/empatheticDialogue1/results_data/model_output_wt_emo10.txt",args)
