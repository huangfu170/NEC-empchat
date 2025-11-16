import argparse
import collections
import logging
import os.path
from datetime import datetime

import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from data.util import compute_metrics


def validate(model, test_dataloader, tokenizer, device, args, epoch):
    print("Validating at epoch {}...".format(epoch))
    model.eval()
    preds = []
    responses = []
    ppls = []
    losses = []
    pbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    
    with torch.no_grad():
        # 使用字典生成式批量处理tensor到device的转换，与train_validate保持一致
        tensor_keys = ['context', 'response', 'social_knowledge', 'emotion']
        for idx, data in pbar:
            tensor_data = {key: data[key].to(device) if data[key] is not None else None 
                          for key in tensor_keys}
            
            # 解包到变量，与train_validate保持一致
            context, response, social_knowledge, emotion = tensor_data['context'], tensor_data['response'], tensor_data['social_knowledge'], tensor_data['emotion']
            
            model.is_training = False
            outputs = model(
                input_ids=context['input_ids'],
                attention_mask=context['attention_mask'],
                labels=response['input_ids'],
                social_knowledge=social_knowledge,
                emotion=emotion,
                social_knowledge_mask=(social_knowledge != tokenizer.pad_token_id).float() if social_knowledge is not None else None,
            )
            mask = (response['input_ids'] != tokenizer.pad_token_id).float().sum()
            all_tokens = response['input_ids'].size(1)*response['input_ids'].size(0)
            ppl = torch.exp((outputs['masked_lm_loss'].detach()/all_tokens*mask).cpu()).item()
            loss = outputs['masked_lm_loss'].detach().cpu().item()
            ppls.append(ppl)
            losses.append(loss)
            
            # 生成预测结果 - 使用与train_validate相同的参数
            pred = model.generate(
                input_ids=context['input_ids'], 
                attention_mask=context['attention_mask'], 
                social_knowledge=social_knowledge, 
                args=args
            )
            preds.extend(pred)
            responses.extend([response['input_ids'][i, :] for i in range(response['input_ids'].size(0))])
            
        model.is_training = True
        _response = pad_sequence(responses, batch_first=True, padding_value=tokenizer.pad_token_id)
        res = compute_metrics((preds, _response.cpu().detach()), 'bart-base', 1, epoch)
        res.update({"ppl": torch.mean(torch.tensor(ppls))})
        return res


def train_validate(model, optimizer, train_loader, valid_loader, tokenizer, device, scheduler, args, train_step=0, epoch=0):
    model.train()
    max_valid_bleu = 0
    
    # Generate high frequency tokens based on args.high_fre_nega
    if args.high_freq_nega:
        high_freq_sentences = get_top(args.CL_sample_num, train_loader)
        high_freq_tokens = tokenizer(high_freq_sentences, padding=True, return_tensors='pt')[
            'input_ids'].to(device)
    else:
        high_freq_tokens = None
    
    # 预先构建emotion到responses的映射，避免每个batch都重新遍历dataset
    emotion_to_responses = None
    if args.emotion_nega:
        emotion_to_responses = build_emotion_to_responses_mapping(train_loader)
    
    for i in range(epoch):
        print(f'start to train the model at epoch {i + 1}................')
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc="Epoch {}".format(i))
        tensor_keys = ['context', 'response', 'social_knowledge', 'emotion']
        for idx, data in pbar:
            model.train()
            # 使用字典生成式批量处理tensor到device的转换，同时处理可能为None的情况
            tensor_data = {key: data[key].to(device) if data[key] is not None else None 
                          for key in tensor_keys}
            
            # 解包到变量
            context, response, social_knowledge, emotion = tensor_data['context'], tensor_data['response'], tensor_data['social_knowledge'], tensor_data['emotion']
            if args.emotion_nega:
                emotion_nega = generate_diff_emotion(emotion_to_responses, emotion, tokenizer, args.CL_sample_num)
            else:
                emotion_nega = None


            outputs = model(
                input_ids=context['input_ids'],
                attention_mask=context['attention_mask'],
                labels=response['input_ids'],
                social_knowledge=social_knowledge,
                emotion=emotion,
                social_knowledge_mask=(social_knowledge != tokenizer.pad_token_id).float() if social_knowledge is not None else None,
                emotion_nega=emotion_nega,
                high_freq=high_freq_tokens,
            )
            optimizer.zero_grad()
            r_loss, cl_loss = outputs['masked_lm_loss'], outputs["cl_loss"]
            loss = r_loss + cl_loss
            loss.backward()
            optimizer.step()
            # scheduler.step()
            pbar.set_description(f"rloss:{r_loss.cpu().item():.4f} cl_loss:{cl_loss.cpu().item():.4f}")
            pbar.update(1)

        if i >= 1:
            res = validate(model, valid_loader, tokenizer, device, args, epoch=i)
            if res['bleu4'] >= max_valid_bleu:
                max_valid_bleu = max(res['bleu4'], max_valid_bleu)
                logging.info(
                    f"VALID: Epoch = {i + 1:d} | BLEU-1: {res['bleu1']} | BLEU-2: {res['bleu2']} | BLEU-3: {res['bleu3']} | BLEU-4: {res['bleu4']}"
                    f"| D-1 / D-2= {'/'.join([str(round(i * 100, 3)) for i in res['macro-distinct']])} "
                    f"| PPL = {res['ppl']}"
                )
                # torch.save(model.state_dict(),
                #            f'DCKS-CL_gen_model_tbs{args.train_beam_size_for_CL}_bs{args.batch_size}_with_emotionshff.pt')
        train_step += 1


def build_emotion_to_responses_mapping(train_loader):
    """
    预先构建emotion到responses的映射，只需要执行一次
    
    Args:
        train_loader: 训练数据加载器
    
    Returns:
        emotion_to_responses: dict, emotion到response列表的映射
    """
    from collections import defaultdict
    
    dataset = train_loader.dataset
    all_emotions = dataset.data['emotion']
    all_responses = dataset.data['response']
    
    emotion_to_responses = defaultdict(list)
    for emotion, response in zip(all_emotions, all_responses):
        emotion_to_responses[int(emotion)].append(response)
    
    return emotion_to_responses


def generate_diff_emotion(emotion_to_responses, emotions, tokenizer, sample_num=3):
    """
    为每个batch内的样例找到指定数量的不同emotion的response
    
    Args:
        emotion_to_responses: dict, 预构建的emotion到response列表的映射
        emotions: 当前batch的emotion tensor [batch_size]
        tokenizer: tokenizer
        sample_num: 每个样例需要的不同emotion response数量，默认3
    
    Returns:
        flattened_responses: 填充后的不同emotion response tensor
    """
    import random
    
    current_emotions = emotions.tolist()
    diff_emotion_responses = []
    
    for current_emotion in current_emotions:
        current_emotion = int(current_emotion)
        sampled_responses = []
        
        # 获取所有不同的emotion类型
        available_emotions = [e for e in emotion_to_responses.keys() if e != current_emotion]
        
        # 从不同emotion中随机采样
        for _ in range(sample_num):
            # 随机选择一个不同的emotion
            chosen_emotion = random.choice(available_emotions)
            # 从该emotion的response中随机选择一个
            chosen_response = random.choice(emotion_to_responses[chosen_emotion])
            sampled_responses.append(chosen_response)
        
        diff_emotion_responses.append(sampled_responses)
    
    flattened_responses = [response for responses in diff_emotion_responses for response in responses]
    flattened_responses = tokenizer(flattened_responses, padding=True, return_tensors='pt')['input_ids'].reshape(emotions.size(0), sample_num, -1)
    return flattened_responses


def get_top(k,train_loader):
    dataset = train_loader.dataset
    data = [res for res in dataset.data['response']]
    new_data = []
    import re
    def simple_word_tokenize(text):
        """
        使用正则表达式进行基础分词
        """
        # 匹配单词字符（字母、数字、下划线）
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    for r in data:
        r = re.split(",|\?|!|\.", r)
        r = [item.strip() for item in r if len(item.strip()) > 0]
        r = [item for item in r if len(simple_word_tokenize(item)) > 1]
        new_data.extend(r)
    counter = collections.Counter(new_data)
    topk_sentence = counter.most_common(k)
    topk_sentence = [item[0] for item in topk_sentence]
    return topk_sentence
