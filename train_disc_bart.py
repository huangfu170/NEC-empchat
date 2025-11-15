import argparse
import logging
import os.path
from datetime import datetime

import torch
import tqdm
from torch import logsumexp
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import BartTokenizer, LogitsProcessorList, NoRepeatNGramLogitsProcessor, StoppingCriteriaList, \
    MaxLengthCriteria

from config import CFG
from empchat.datasets import empchat
from empchat.util import unzip_batch, compute_metrics, seed_everything,get_collate_fn,test_beam_search
from question_seperate.model import QuestionFixedModel, QuestionFixedModel2,QuestionFixedModel3,QuestionFixedModel4


def validate(model, test_dataloader, tokenizer, device, epoch):
    print("Validating at epoch {}...".format(epoch))
    preds = []
    _responses = []
    ppls = []
    accs = []
    losses = []
    pbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    with torch.no_grad():
        for idx, data in pbar:
            model.eval()
            _msk = data['attention_mask'].to(device)
            _input, _response = data['input_ids'].to(device), data['response'].to(device)
            _question = data['question'].to(device)
            _emotion=data['emotion'].to(device)
            # loss, r_logits = model(_input, _msk, _response, _question)
            (rloss, mloss), (information_missing_score, acc), r_logits = model(_input, _msk, _response, _question)
            # (rloss,mloss,emo_loss), (information_missing_score, acc,emo_acc), r_logits = model(_input, _msk, _response, _question,_emotion)
            ppl = torch.exp(rloss.detach().cpu()).item()
            loss = (rloss+0.5*mloss).detach().cpu().item()
            # ppl=torch.exp(loss).cpu().item()
            ppls.append(ppl)
            # losses.append(loss.cpu().item())
            losses.append(loss)
            accs.append(acc.cpu().item())
            pbar.set_description(f"loss:{loss:.4f} ppl:{ppl:.4f} acc:{acc:4f}")
            model.is_training=False
            pred = test_beam_search(model,_input,_msk,_question).tolist()
            model.is_training=True
            preds.extend(pred)
            _responses.extend([_response[i, :] for i in range(_response.size(0))])
        _response = pad_sequence(_responses, batch_first=True, padding_value=tokenizer.pad_token_id)
        res= compute_metrics((preds, _response.cpu().detach()), 'bart-base', 1, epoch)
        pad_pred = pad_sequence([torch.tensor(t) for t in preds], batch_first=True,
                                padding_value=tokenizer.pad_token_id)
        print(pad_pred[:10,:])
        print(tokenizer.batch_decode(sequences=pad_pred[:10,:], skip_special_tokens=True)[:10])
        res['ppl']=torch.mean(torch.tensor(ppls))
        if len(accs) != 0:
            res['acc']=torch.mean(torch.tensor(accs))
            return res
        else:
            return res


def train_validate(
        model,
        optimizer,
        train_loader,
        valid_loader,
        tokenizer,
        device,
        scheduler,
        train_step=0,
        epoch=0
):
    model.train()
    q_crit = torch.nn.CrossEntropyLoss()
    max_train_bleu = 0
    max_valid_bleu = 0
    min_ppl = 1e7

    for i in tqdm.trange(epoch):
        for idx, data in enumerate(train_loader):
            model.train()
            _msk = data['attention_mask'].to(device)
            _input, _response = data['input_ids'].to(device), data['response'].to(device)
            _question = data['question'].to(device)
            _emotion=data['emotion'].to(device)
            # loss, r_logits = model(_input, _msk, _response)
            (rloss,mloss), _, r_logits = model(_input, _msk, _response, _question)
            # (rloss,mloss,emo_loss), _, r_logits = model(_input, _msk, _response, _question,_emotion)
            loss=rloss+1.5*mloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            if i >= 10 and idx % 500 == 0:

                # pred = greedy_search(model, _input, _msk, _question, tokenizer.eos_token_id, tokenizer.bos_token_id, max_len=20)
                # pred = batch_beam_search(model, r_logits, _input, _msk, _question, 5, tokenizer.eos_token_id,
                #                          2,
                #                          3,
                #                          max_len=30)

                res = compute_metrics((pred, _response), 'bart-base', 1, i,False)
                if res['bleu4'][-1] > max_train_bleu:
                    max_train_bleu = bleu['precisions'][-1]
                    logging.info(
                        f"TRAIN: Epoch = {i + 1:d} | BLEU 1-4: = {res['bleu1'] * 100:.2f} | {res['bleu2'] * 100:.2f} | "
                        f"{res['bleu3'] * 100:.2f} | {res['bleu4'] * 100:.2f} % "
                        f"| Distinct = {[round(i * 100, 2) for i in res['macro-distinct']]} "
                    )
                # pad_pred = pad_sequence([torch.tensor(t) for t in pred], batch_first=True,
                #                         padding_value=tokenizer.pad_token_id)
                # print(tokenizer.batch_decode(sequences=pad_pred, skip_special_tokens=True))
        print(f'start to train the model at epoch {i}................')
        if i % 1 == 0:
            res= validate(model, valid_loader, tokenizer, device, epoch=i)
            # valid_bleu, valid_distinct, ppl = validate(model, valid_loader, tokenizer, device, epoch=i)
            if res['ppl'] < min_ppl:
                min_ppl = res['ppl']
                max_valid_bleu = max(res['bleu4'], max_valid_bleu)
                logging.info(
                    f"VALID: Epoch = {i + 1:d} | BLEU 1-4: = {res['bleu1'] * 100:.2f} | {res['bleu2'] * 100:.2f} | "
                    f"{res['bleu3'] * 100:.2f} | {res['bleu4'] * 100:.2f} % "
                    f"| Distinct = {[round(i * 100, 2) for i in res['macro-distinct']]} "
                    f"| PPL = {res['ppl']} | Acc={res['acc']}"
                )

        train_step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda-id', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()
    cfg = CFG()

    print("正在使用{}".format("计算机学�?" if cfg.isCU else "曙光"))
    logging.basicConfig(level=logging.INFO, filename=datetime.now().strftime(
        '%Y-%m-%d') + '-mloss_真正的linear分离+双encoder+增强分辨' + '.log')
    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    seed_everything(42)
    tokenizer = BartTokenizer.from_pretrained(os.path.join(cfg.data_prefix, 'bart-base/'))
    tokenizer.add_tokens(['[MISS]'])
    model = QuestionFixedModel3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95)
    dataset = empchat.EmpDataset('bart-base', 'train', os.path.join(cfg.data_prefix, 'empatheticdialogues/'), history_len=10,use_comet=False,use_emotion=True)
    dataloader = DataLoader(dataset, 16, shuffle=False, collate_fn=get_collate_fn(tokenizer.pad_token_id))

    valid_dataset = empchat.EmpDataset('bart-base', 'test', os.path.join(cfg.data_prefix, 'empatheticdialogues/'),
                                       history_len=10,use_comet=False,use_emotion=True)
    valid_dataloader = DataLoader(valid_dataset, 32, shuffle=False, collate_fn=get_collate_fn(tokenizer.pad_token_id))
    train_validate(model, optimizer, dataloader, valid_dataloader, tokenizer, device, None
                   , 0, args.epoch)
    # validate(model, valid_dataloader, tokenizer, device, epoch=0)
