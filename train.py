import collections
import random
import re

import swanlab
import torch
import torch.distributed as dist
import tqdm
from torch.nn.utils.rnn import pad_sequence

from data.util import compute_metrics


def _get_module(model):
    """Get the underlying module from DDP wrapper."""
    return model.module if hasattr(model, 'module') else model


def validate(model, test_dataloader, tokenizer, device, args, epoch, rank=0, world_size=1):
    if rank == 0:
        print(f"Validating at epoch {epoch}...")
    model.eval()
    raw_model = _get_module(model)
    preds, responses, ppls = [], [], []
    tensor_keys = ['context', 'response', 'social_knowledge', 'emotion']

    with torch.no_grad():
        iterator = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)) if rank == 0 else enumerate(test_dataloader)
        for _, data in iterator:
            batch = {k: data[k].to(device) if data[k] is not None else None for k in tensor_keys}
            context, response, social_knowledge, emotion = (
                batch['context'], batch['response'], batch['social_knowledge'], batch['emotion']
            )
            sk_mask = (social_knowledge != tokenizer.pad_token_id).float() if social_knowledge is not None else None

            raw_model.is_training = False
            outputs = raw_model(
                input_ids=context['input_ids'],
                attention_mask=context['attention_mask'],
                labels=response['input_ids'],
                social_knowledge=social_knowledge,
                emotion=emotion,
                social_knowledge_mask=sk_mask,
            )

            n_valid_tokens = (response['input_ids'] != tokenizer.pad_token_id).float().sum()
            n_total_tokens = response['input_ids'].numel()
            avg_loss = outputs['masked_lm_loss'].detach() * n_total_tokens / n_valid_tokens
            ppls.append(torch.exp(avg_loss.cpu()).item())

            pred = raw_model.generate(
                input_ids=context['input_ids'],
                attention_mask=context['attention_mask'],
                social_knowledge=social_knowledge,
                social_knowledge_mask=sk_mask,
                args=args,
            )
            preds.extend(pred)
            responses.extend(response['input_ids'][i] for i in range(response['input_ids'].size(0)))

    raw_model.is_training = True

    # Gather results from all ranks
    if world_size > 1:
        all_preds = [None] * world_size
        all_responses = [None] * world_size
        all_ppls = [None] * world_size
        dist.all_gather_object(all_preds, [p.cpu() if isinstance(p, torch.Tensor) else p for p in preds])
        dist.all_gather_object(all_responses, [r.cpu() for r in responses])
        dist.all_gather_object(all_ppls, ppls)
        if rank == 0:
            preds = [p for rank_preds in all_preds for p in rank_preds]
            responses = [r for rank_responses in all_responses for r in rank_responses]
            ppls = [p for rank_ppls in all_ppls for p in rank_ppls]

    if rank == 0:
        padded_responses = pad_sequence(responses, batch_first=True, padding_value=tokenizer.pad_token_id)
        metrics = compute_metrics((preds, padded_responses.cpu().detach()), tokenizer, 1, epoch, output_dir=args.model_save_folder)
        metrics['ppl'] = torch.mean(torch.tensor(ppls))
        return metrics
    return None


def train_validate(model, optimizer, train_loader, valid_loader, tokenizer, device, scheduler, args, train_step=0, epoch=0, rank=0, world_size=1):
    model.train()
    raw_model = _get_module(model)
    max_valid_bleu = 0
    tensor_keys = ['context', 'response', 'social_knowledge', 'emotion']

    high_freq_tokens = None
    if args.high_freq_nega:
        sentences = get_high_freq_sentences(args.CL_sample_num, train_loader)
        high_freq_tokens = tokenizer(sentences, padding=True, return_tensors='pt')['input_ids'].to(device)

    emotion_to_responses = build_emotion_to_responses(train_loader) if args.emotion_nega else None

    for i in range(epoch):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(i)
        if rank == 0:
            print(f'Epoch {i + 1} training started...')
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {i + 1}") if rank == 0 else enumerate(train_loader)

        for _, data in pbar:
            model.train()
            batch = {k: data[k].to(device) if data[k] is not None else None for k in tensor_keys}
            context, response, social_knowledge, emotion = (
                batch['context'], batch['response'], batch['social_knowledge'], batch['emotion']
            )
            sk_mask = (social_knowledge != tokenizer.pad_token_id).float() if social_knowledge is not None else None
            emotion_nega = sample_diff_emotion_responses(emotion_to_responses, emotion, tokenizer, args.CL_sample_num) if args.emotion_nega else None

            outputs = model(
                input_ids=context['input_ids'],
                attention_mask=context['attention_mask'],
                labels=response['input_ids'],
                social_knowledge=social_knowledge,
                emotion=emotion,
                social_knowledge_mask=sk_mask,
                emotion_nega=emotion_nega,
                high_freq=high_freq_tokens,
            )

            r_loss = outputs['masked_lm_loss']
            cl_loss = outputs['cl_loss']
            optimizer.zero_grad()
            (r_loss + cl_loss).backward()
            optimizer.step()
            if rank == 0:
                pbar.set_description(f"r_loss:{r_loss.item():.4f} cl_loss:{cl_loss.item():.4f}")
                swanlab.log({
                    "train/r_loss": r_loss.item(),
                    "train/cl_loss": cl_loss.item(),
                    "train/total_loss": (r_loss + cl_loss).item(),
                })

        if i >= 1:
            if hasattr(valid_loader.sampler, 'set_epoch'):
                valid_loader.sampler.set_epoch(i)
            res = validate(model, valid_loader, tokenizer, device, args, epoch=i, rank=rank, world_size=world_size)
            if rank == 0:
                swanlab.log({
                    "epoch": i + 1,
                    "bleu1": res['bleu1'], "bleu2": res['bleu2'],
                    "bleu3": res['bleu3'], "bleu4": res['bleu4'],
                    "distinct1": res['macro-distinct'][0] * 100,
                    "distinct2": res['macro-distinct'][1] * 100,
                    "ppl": res['ppl'],
                })
                if res['bleu4'] >= max_valid_bleu:
                    max_valid_bleu = res['bleu4']
                    save_path = f"{args.model_save_folder}/best_model"
                    raw_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"Best model saved to {save_path} (bleu4={res['bleu4']:.4f})")
            if world_size > 1:
                dist.barrier()

        train_step += 1


def build_emotion_to_responses(train_loader):
    """Build a mapping from emotion label to list of response strings."""
    emotion_map = collections.defaultdict(list)
    for emotion, response in zip(train_loader.dataset.data['emotion'], train_loader.dataset.data['response']):
        emotion_map[int(emotion)].append(response)
    return emotion_map


def sample_diff_emotion_responses(emotion_to_responses, emotions, tokenizer, sample_num=3):
    """For each sample in the batch, sample `sample_num` responses from different emotion classes."""
    results = []
    for emotion in emotions.tolist():
        emotion = int(emotion)
        other_emotions = [e for e in emotion_to_responses if e != emotion]
        sampled = [
            random.choice(emotion_to_responses[random.choice(other_emotions)])
            for _ in range(sample_num)
        ]
        results.append(sampled)

    flat = [r for group in results for r in group]
    return tokenizer(flat, padding=True, return_tensors='pt')['input_ids'].reshape(emotions.size(0), sample_num, -1)


def _tokenize_words(text):
    return re.findall(r'\b\w+\b', text.lower())


def get_high_freq_sentences(k, train_loader):
    """Return the top-k most frequent sub-sentences from training responses."""
    sentences = []
    for response in train_loader.dataset.data['response']:
        for segment in re.split(r'[,?!.]', response):
            segment = segment.strip()
            if segment and len(_tokenize_words(segment)) > 1:
                sentences.append(segment)
    return [s for s, _ in collections.Counter(sentences).most_common(k)]
