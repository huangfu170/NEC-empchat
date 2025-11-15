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
from empchat.util import unzip_batch, compute_metrics, seed_everything, get_collate_fn
from question_seperate.model import QuestionFixedModel, QuestionFixedModel2



def greedy_search(model, _input, _msk, _question, eos_token_id, bos_token_id, max_len=20):
    bs = _input.size(0)
    device = _input.device
    end_tokens = [False for _ in range(bs)]
    res = torch.ones(bs, 1, dtype=torch.long, device=device) * bos_token_id
    pred_outputs = torch.ones(bs, _input.size(-1) + max_len, dtype=torch.long, device=device) * eos_token_id
    pred_outputs[:, :_input.size(-1)] = _input
    with torch.no_grad():
        for k in range(max_len):
            logits = model(_input, _msk, res, _question)[-1]
            logits = logits[:, -1, :]
            pred = torch.argmax(logits, dim=-1)
            for b in range(bs):
                if pred[b].item() == eos_token_id:
                    end_tokens[b] = True
                if not end_tokens[b]:
                    pred_outputs[b, _input.size(-1) + k] = pred[b]

            res = torch.cat([res, pred.unsqueeze(1)], dim=-1)

    return pred_outputs


def _n_gram_overlap(seq, n_gram):
    if len(seq) <= n_gram + 1:
        return False
    seq_str = [str(t) for t in seq[1:]]
    n_gram_set = set()
    for i in range(len(seq_str) - n_gram + 1):
        n_gram_str = ','.join(seq_str[i:(i + n_gram)])
        if not n_gram_str in n_gram_set:
            n_gram_set.add(n_gram_str)
        else:
            return True
    return False


def _sort_by_log_probs(result_seqs, log_probs, alpha):
    seq_lens = torch.tensor([len(seq) - 1 for seq in result_seqs])  # bos不算
    len_norm = seq_lens ** alpha
    sorted_id = (log_probs.detach().cpu() / len_norm).argsort(descending=True)
    new_log_probs = (log_probs.detach().cpu() / len_norm)[sorted_id]
    new_result_seqs = [result_seqs[i] for i in sorted_id]
    return new_result_seqs, new_log_probs


def _reorganize(result_seqs, log_probs, EOS_ID, n_gram):
    """Reorganize result_seqs so that finished sequences are at the end.
    """
    continue_id = []
    finished_id = []
    continue_seqs = []
    finished_seqs = []

    for i, seq in enumerate(result_seqs):
        if seq[-1] == EOS_ID:
            finished_id.append(i)
            finished_seqs.append(seq)
        else:
            continue_id.append(i)
            continue_seqs.append(seq)
            if _n_gram_overlap(seq, n_gram):
                log_probs[i] = -1e20
    new_beam_width = len(continue_id)
    reordered_id = continue_id + finished_id
    result_seqs = continue_seqs + finished_seqs

    return result_seqs, log_probs[reordered_id], new_beam_width


# TODO: 需要预测beam_search中的Perplexity
def batch_beam_search(model, logits, _input, _msk, _question, beam_width, eos_token_id, bos_token_id, n_gram,
                      max_len=20, return_sequences_num=1):
    bs = logits.size(0)
    model.eval()
    assert bs >= 1
    ress = []
    for i in range(bs):
        res, log_probs = beam_search(model, logits[i].unsqueeze(0), _input[i].unsqueeze(0), _msk[i].unsqueeze(0),
                                     _question[i].unsqueeze(0), beam_width, eos_token_id, bos_token_id, n_gram, max_len)

        ress.extend(res[:return_sequences_num])
    model.train()
    return ress


def beam_search(model, logits, _input, _msk, _question, beam_width, eos_token_id, bos_token_id, n_gram, max_len=20,
                alpha=0.1):
    from transformers import BeamSearchScorer
    bs = logits.size(0)
    device = logits.device
    res = torch.ones(bs, 1, dtype=torch.long, device=device) * bos_token_id
    # 这里要注意的是，如果出现eos，也就意味着在该条路径上搜索结束，同时beam_width也要减少
    # 如果beam_width不减少，那么就会出现搜索路径数永远都是beam_width的情况
    # 这就意味着，我们至少进行了vocab_size-width次搜索

    with torch.no_grad():
        outputs = model(_input, _msk, res, _question)
        logits = outputs[-1]
        loss = outputs[0]
        logits = logits[:, -1, :]
        logits = logits.reshape(-1)  # (vocab_size,)
    # 在第一轮的时候，logits的shape是(bs, vocab_size*beam_size)
    pred = torch.topk(logits, beam_width, dim=-1)  # pred的两部分的shape都是(bs, beam_width)
    vocab_size = logits.size(-1)

    pred_indices = pred.indices.view(-1).tolist()
    result_seqs = [[bos_token_id, i] for i in pred_indices]
    log_probs = pred.values - logsumexp(logits, dim=-1, keepdim=True)  # 对数概率（等价于对softmax的概率求对数）

    result_seqs, log_probs, new_beam_width = _reorganize(result_seqs, log_probs, bos_token_id, n_gram)
    # log_probs的shape是
    for k in range(max_len):
        if new_beam_width == 0:
            break

        res = torch.tensor(result_seqs[:new_beam_width]).to(device)
        # TODO: 将loss存起来，和log_probs同存同取
        with torch.no_grad():
            logits = model(_input, _msk, res, _question)[-1]
            logits = logits[:, -1, :]  # (new_beam_width, vocab_size)
        reshaped_logits = logits.reshape(-1)  # (new_beam_width*vocab_size,)
        pred = torch.topk(reshaped_logits, new_beam_width, dim=-1)

        current_ids = pred.indices  # 词在vocab里的id
        beam_id = current_ids // vocab_size
        vocab_id = current_ids % vocab_size

        current_log_probs = pred.values.reshape(new_beam_width, -1) - logsumexp(logits, dim=-1, keepdim=True)
        log_probs[:new_beam_width] = current_log_probs.reshape(-1)

        new_seqs = []
        for beam, token in zip(beam_id, vocab_id):
            new_seqs.append(result_seqs[beam] + [token.item()])
        result_seqs[:new_beam_width] = new_seqs

        result_seqs, log_probs, new_beam_width = _reorganize(result_seqs, log_probs, bos_token_id, n_gram)

    if new_beam_width > 0:
        res = torch.tensor(result_seqs[:new_beam_width]).to(device)
        logits = model(_input, _msk, res, _question)[-1]
        eos_log_probs = logits[:, -1, eos_token_id] - logsumexp(logits[:, -1, eos_token_id], dim=-1)
        log_probs[:new_beam_width] += eos_log_probs
        for i in range(new_beam_width):
            result_seqs[i].append(eos_token_id)

    result_seqs, log_probs = _sort_by_log_probs(result_seqs, log_probs, alpha)
    return result_seqs, log_probs





def validate(model, test_dataloader, tokenizer, device, epoch):
    print("Validating at epoch {}...".format(epoch))
    model.eval()
    preds = []
    _responses = []
    ppls = []
    accs = []
    losses = []
    pbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    with torch.no_grad():
        for idx, data in pbar:
            params = unzip_batch(data, device)
            _msk = params[-1]
            _input, _response = params[0], params[2]
            _question = params[3]

            loss, r_logits = model(_input, _msk, _response, _question)
            # (rloss, mloss), (information_missing_score, acc), r_logits = model(_input, _msk, _response, _question)
            # ppl = torch.exp(rloss.detach().cpu()).item()
            # loss = (rloss+0.5*mloss).detach().cpu().item()

            ppl=torch.exp(loss).cpu().item()
            ppls.append(ppl)
            losses.append(loss.cpu().item())
            losses.append(loss)
            # accs.append(acc.cpu().item())
            pbar.set_description(f"loss:{loss:.4f} ppl:{ppl:.4f}")
            pred = batch_beam_search(model, r_logits, _input, _msk, _question, 10, tokenizer.eos_token_id,
                                     tokenizer.bos_token_id,
                                     3,
                                     max_len=20)
            preds.extend(pred)
            _responses.extend([_response[i, :] for i in range(_response.size(0))])
        _response = pad_sequence(_responses, batch_first=True, padding_value=tokenizer.pad_token_id)

        bleu, distinct = compute_metrics((torch.tensor(preds), _response.cpu().detach()), 'bart-base', 1, epoch)
        # logging.info(
        #     f"Test: Epoch = {epoch + 1:d} | BLEU = {bleu['bleu'] * 100:.5f} % | BLEU-1~4: {bleu['precisions']}"
        #     f"| Distinct = {[round(i * 100, 5) for i in distinct['micro-distinct']]} "
        # )
        pad_pred = pad_sequence([torch.tensor(t) for t in preds], batch_first=True,
                                padding_value=tokenizer.pad_token_id)
        print(tokenizer.batch_decode(sequences=pad_pred, skip_special_tokens=True)[:10])
        if len(accs) != 0:
            return bleu, distinct, torch.mean(torch.tensor(ppls)), torch.mean(torch.tensor(accs))
        else:
            return bleu, distinct, torch.mean(torch.tensor(ppls))


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
            params = unzip_batch(data, device)
            _msk = params[-1]
            _input, _response = params[0], params[2]
            _question = params[3]

            loss, r_logits = model(_input, _msk, _response)
            # (rloss,mloss), _, r_logits = model(_input, _msk, _response, _question)
            # loss=rloss+0.5*mloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            if i >= 10 and idx % 500 == 0:

                # pred = greedy_search(model, _input, _msk, _question, tokenizer.eos_token_id, tokenizer.bos_token_id, max_len=20)
                pred = batch_beam_search(model, r_logits, _input, _msk, _question, 10, tokenizer.eos_token_id,
                                         tokenizer.bos_token_id,
                                         3,
                                         max_len=30)
                bleu, distinct = compute_metrics((torch.tensor(pred), _response), 'bart-base', 1, i,False)
                if bleu['precisions'][-1] > max_train_bleu:
                    max_train_bleu = bleu['precisions'][-1]
                    logging.info(
                        f"TRAIN: Epoch = {i + 1:d} | BLEU = {bleu['bleu'] * 100:.5f} % | BLEU-1~4: {bleu['precisions']}"
                        f"| Distinct = {[round(i * 100, 5) for i in distinct['micro-distinct']]} "
                    )
                # pad_pred = pad_sequence([torch.tensor(t) for t in pred], batch_first=True,
                #                         padding_value=tokenizer.pad_token_id)
                # print(tokenizer.batch_decode(sequences=pad_pred, skip_special_tokens=True))
        print(f'start to train the model at epoch {i}................')
        if i > 10 and i % 4 == 0:
            # valid_bleu, valid_distinct, ppl, acc = validate(model, valid_loader, tokenizer, device, epoch=i)
            valid_bleu, valid_distinct, ppl = validate(model, valid_loader, tokenizer, device, epoch=i)
            if ppl < min_ppl:
                min_ppl = ppl
                max_valid_bleu = max(valid_bleu['precisions'][-1], max_valid_bleu)
                # logging.info(
                #     f"VALID: Epoch = {i + 1:d} | BLEU = {valid_bleu['bleu'] * 100:.5f} % | BLEU-1~4: {valid_bleu['precisions']}"
                #     f"| Distinct = {[round(i * 100, 5) for i in valid_distinct['micro-distinct']]} "
                #     f"| PPL = {ppl} | Acc={acc}"
                # )
                logging.info(
                    f"VALID: Epoch = {i + 1:d} | BLEU = {valid_bleu['bleu'] * 100:.5f} % | BLEU-1~4: {valid_bleu['precisions']}"
                    f"| Distinct = {[round(i * 100, 5) for i in valid_distinct['micro-distinct']]} "
                    f"| PPL = {ppl} "
                )
        train_step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda-id', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=50)
    args = parser.parse_args()
    cfg = CFG()

    print("正在使用{}".format("计算机学院" if cfg.isCU else "曙光"))
    logging.basicConfig(level=logging.INFO, filename=datetime.now().strftime(
        '%Y-%m-%d') + '-raw_bart' + '.log')
    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    seed_everything(42)
    tokenizer = BartTokenizer.from_pretrained(os.path.join(cfg.data_prefix, 'bart-base/'))
    tokenizer.add_tokens(['[MISS]'])
    model = QuestionFixedModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95)
    dataset = empchat.EmpDataset('bert', 'train', os.path.join(cfg.data_prefix, 'empatheticdialogues/'), history_len=10)
    dataloader = DataLoader(dataset, 16, shuffle=False, collate_fn=get_collate_fn(tokenizer.pad_token_id))

    valid_dataset = empchat.EmpDataset('bert', 'test', os.path.join(cfg.data_prefix, 'empatheticdialogues/'),
                                       history_len=10)
    valid_dataloader = DataLoader(valid_dataset, 64, shuffle=False, collate_fn=get_collate_fn(tokenizer.pad_token_id))
    train_validate(model, optimizer, dataloader, valid_dataloader, tokenizer, device, None
                   , 0, args.epoch)
    # validate(model, valid_dataloader, tokenizer, device, epoch=0)
