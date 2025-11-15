import argparse
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from config import CFG
from data.datasets.empchat import RankingDataset
from data.util import seed_everything
from question_seperate.model import BertForRanking


def collate_fn(batch):
    p_input_ids = [pt for item in batch if item is not None for pt in item["p_input_ids"]]
    n_input_ids=[nt for item in batch if item is not None for nt in item['n_input_ids']]
    if isinstance(n_input_ids[0],list):
        n_input_ids=[item for sublist in n_input_ids for item in sublist]
    p_label=[1]*len(p_input_ids)
    n_label=[-1]*len(n_input_ids)

    input_ids=p_input_ids+n_input_ids
    label=p_label+n_label
    t=list(range(len(input_ids)))
    batch=list(zip(input_ids,label,t))
    # random.shuffle(batch)
    input_ids,label,t=zip(*batch)
    input_ids=pad_sequence(input_ids,batch_first=True,padding_value=pad_token_id)
    label=torch.tensor(label)
    return {
        'input_ids': input_ids,
        'labels': label,
        'shuffle_seed': t,
    }
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cuda-id', type=int, default=1)
    parser.add_argument('--rand-seed', type=int, default=42)
    args=parser.parse_args()

    seed_everything(args.rand_seed)
    device=torch.device('cuda:{}'.format(args.cuda_id)) if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = RankingDataset('/home/huangfu/empdialogue_code/empatheticDialogue1/config.yaml','train')
    test_dataset = RankingDataset('/home/huangfu/empdialogue_code/empatheticDialogue1/config.yaml','test')
    pad_token_id = 0

    train_dataloader= torch.utils.data.DataLoader(train_dataset, batch_size=72, shuffle=True, collate_fn=collate_fn)
    test_dataloader= torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    model=BertForRanking.from_pretrained("/home/huangfu/empdialogue_code/empatheticDialogue1/model_checkpoint/bert-base")
    model.bert.resize_token_embeddings(len(train_dataset.tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_func=nn.MarginRankingLoss(margin=0.3)
    cross_entropy=nn.CrossEntropyLoss()
    schechuler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=len(train_dataloader),gamma=0.5)
    min_loss=1e5
    for epoch in range(10):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for idx, data in pbar:

            model.train()
            pos_data=data['input_ids'][:data['input_ids'].size(0)//2][:16]
            neg_data=data['input_ids'][data['input_ids'].size(0)//2:][:16]
            pos_margin_logits,pos_logits=model(pos_data.to(device),((pos_data!=0).float()).to(device))
            neg_margin_logits,neg_logits=model(neg_data.to(device),((neg_data!=0).float()).to(device))
            label=torch.tensor([1]*pos_margin_logits.size(0)+[0]*neg_margin_logits.size(0))


            t=sorted(list(zip(list(range(0,data['input_ids'].size(0))),data['shuffle_seed'])),key=lambda x:x[1])
            t=[item[0] for item in t]
            optimizer.zero_grad()
            margin_loss=loss_func(pos_margin_logits,neg_margin_logits,torch.ones_like(pos_margin_logits).to(device))
            cls_loss=cross_entropy(torch.cat([pos_logits,neg_logits],dim=0),label.to(device))
            loss=margin_loss+cls_loss
            loss.backward()
            optimizer.step()
            schechuler.step()
            pbar.set_description(f"loss:{loss.cpu().item():.4f},margin_loss:{margin_loss.cpu().item():.4f}")
            pbar.update(1)

        model.eval()
        vbar=tqdm(enumerate(test_dataloader),total=len(test_dataloader))
        with torch.no_grad():
            loss_total=0
            for vidx,vdata in vbar:
                pos_data = vdata['input_ids'][:vdata['input_ids'].size(0) // 2][:128]
                neg_data = vdata['input_ids'][vdata['input_ids'].size(0) // 2:][:128]
                pos_margin_logits, pos_logits = model(pos_data.to(device), (pos_data != 0).float().to(device))
                neg_margin_logits, neg_logits = model(neg_data.to(device), (neg_data != 0).float().to(device))


                loss=loss_func(pos_margin_logits,neg_margin_logits,torch.ones_like(pos_margin_logits).to(device))
                vbar.set_description(f" valid loss:{loss.cpu().item():.4f}")
                vbar.update(1)
                loss_total+=loss.cpu().item()
            print(f"valid loss:{loss_total/len(test_dataloader):.4f}")
            if loss_total/len(test_dataloader)<min_loss:
                min_loss=loss_total/len(test_dataloader)
                model.save_pretrained(f'DCKS-entity_ranking_model_context_{0.3-min_loss}')
                train_dataset.tokenizer.save_pretrained(f'DCKS-entity_ranking_model_context_{0.3-min_loss}')
                # torch.save(model.state_dict(), f'DCKS-entity_ranking_model_context_{0.3-min_loss}.pt')
                print('model saved loss is {}'.format(min_loss))




