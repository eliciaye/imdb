import weightwatcher as ww
import random
from collections import OrderedDict
from importlib import import_module
from operator import itemgetter
import json
import math
import warnings
import argparse
import time
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchtext import data
from torchtext import datasets
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from tqdm import tqdm

from model import *
from utils import *
from transformer import *

# from dataset import *
import ipdb
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--GPU_id', default="0", type=str)
parser.add_argument('--run_name', default='tmp', type=str)
parser.add_argument('--use_weightwatcher',action = 'store_true')
parser.add_argument('--record_alpha', default = False, action = 'store_true')
parser.add_argument('--record_lr', default = False, action = 'store_true')
parser.add_argument('--percentage', type = float, default = 0.3)
parser.add_argument('--epoch_num', type = int, default = 200)
parser.add_argument('--baseline', default = False, action = 'store_true')
parser.add_argument('--method', type = int, default = 1)
parser.add_argument('--random_seed', type = int, default = 1234)
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--nhid', type = int, default = 100)
parser.add_argument('--nemb', type = int, default = 100)
parser.add_argument('--nhead', type = int, default = 10)
parser.add_argument('--nlayers', type = int, default = 4)
parser.add_argument('--dropout', type = float, default = 0.3)
parser.add_argument('--max_sentence_length', type = int, default = 500)
parser.add_argument('--average', type = str, default = '')
parser.add_argument('--sample_evals', action = 'store_true')

# MAX_VOCAB_SIZE = 25_000
MAX_VOCAB_SIZE = 25000
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

best_valid_loss = float('inf')
args = parser.parse_args()

writer = SummaryWriter('./tensorboard_logs/{}'.format(args.run_name))
os.environ["OMP_NUM_THREADS"] = "1"

# os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.random_seed)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  include_lengths = True)

LABEL = data.LabelField(dtype = torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL,root='.data')

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data), 
    batch_size = args.batch_size,
    sort_within_batch = True,
    device = device)
# IMDB_dataset = IMDB(args, device)

INPUT_DIM = len(TEXT.vocab)

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

# print(PAD_IDX)  # 1 
# model = RNN(INPUT_DIM, 
#             EMBEDDING_DIM, 
#             HIDDEN_DIM, 
#             OUTPUT_DIM, 
#             N_LAYERS, 
#             BIDIRECTIONAL, 
#             DROPOUT, 
#             PAD_IDX)

print("transformer size: ", INPUT_DIM, args.nemb, args.nhid, args.nhead, args.nlayers, args.dropout)
# 25002 100 100 10 3 0.3
model = Transformer(INPUT_DIM, args.nemb, args.nhid, args.nhead, args.nlayers, args.dropout)

for name, param in model.named_parameters():
    print(name)
for i,(name, para) in enumerate(model.named_parameters()):
    # print(name)
    if ('weight' in name) and (('qkv_nets' in name) or ('linear' in name) or ('out_projection_net' in name) or ('decoder' in name)):
        print(i,name)

pretrained_embeddings = TEXT.vocab.vectors

# print(pretrained_embeddings.shape)  #[25002, 100]

model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

WARMUP_STEPS=4000

optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=get_custom_lr(0,WARMUP_STEPS,args.nemb))

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion    

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    # ipdb.set_trace()
    model.train()
    # ipdb.set_trace()
    for batch in tqdm(iterator):
        model.zero_grad()
        optimizer.zero_grad()
        text, text_lengths = batch.text         # [Seq, Batch_size]
        if text.shape[0] > args.max_sentence_length:
            text = text[:args.max_sentence_length]
        # if text.shape[1] > args.max_sentence_length:
        #     text = text[:, :args.max_sentence_length]
        text = text.permute(1, 0).contiguous()#.cuda()
        # text = text
        label = batch.label
        # print(text.shape)
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, label)
        
        acc = binary_accuracy(predictions, label)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in tqdm(iterator):
            text, text_lengths = batch.text
            if text.shape[0] > args.max_sentence_length:
                text = text[:args.max_sentence_length]
            
            # if text.shape[1] > args.max_sentence_length:
            #     text = text[:, :args.max_sentence_length]
            text = text.permute(1, 0).contiguous()#.cuda()
            # text = text
            label = batch.label
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

best_test_acc = 0.
for epoch in tqdm(range(args.epoch_num)):
    start_time = time.time()
    # ipdb.set_trace()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    #     torch.save(model.state_dict(), 'tut2-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
    
    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('test_loss', test_loss, global_step=epoch)
    writer.add_scalar('train_acc', train_acc, global_step=epoch)
    writer.add_scalar('test_acc', test_acc, global_step=epoch)
    
    ROOT = f'/data/eliciaye/nlp/{args.run_name}/'
    if not os.path.exists(ROOT):
        os.makedirs(ROOT)
    SAVE_PATH = f'/data/eliciaye/nlp/{args.run_name}/ckpt.pt'
    with open(SAVE_PATH, 'wb') as f:
        torch.save(model, f)
    if test_acc > best_test_acc:
        BEST_PATH = f'/data/eliciaye/nlp/{args.run_name}/model_best.pt'
        with open(BEST_PATH, 'wb') as f:
            torch.save(model, f)
        best_test_acc = test_acc
    
    # optimizer = optim.SGD(model.parameters(), 
    #                         lr = get_lr(epoch, args.lr, args.epoch_num))
    
    if args.use_weightwatcher:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = dict()
            watcher = ww.WeightWatcher(model = model)
            if epoch % 10:
                details = watcher.analyze(vectors=False, plot=True, savefig=ROOT+'/epoch{}'.format(epoch), fix_fingers=False, fit='PL',sample_evals=args.sample_evals)
            else:
                details = watcher.analyze(vectors=False, fix_fingers=False, fit='PL',sample_evals=args.sample_evals)
            details.to_csv(ROOT+'details.csv')
            layers = details['layer_id'].values


        if args.record_alpha:
            writer.add_scalars(main_tag = 'alpha/layer', 
                               tag_scalar_dict={str(i): details.loc[i, 'alpha'] for i in range(len(details))},
                               global_step=epoch)
            
        if args.baseline:
            
            if args.record_lr:
                writer.add_scalars(main_tag='lr/layer',
                                tag_scalar_dict={str(i): optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))},
                                global_step=epoch)
                
        elif args.method == 1:
            para_save = []
            para_copy = []
            net_para1 = []
            alpha_all = []
            n = len(details)
            # print(n)
            print(details)
            for name, para in model.named_parameters():
                # print(name)
                if ('weight' in name) and (('qkv_nets' in name) or ('linear' in name) or ('out_projection_net' in name) or ('decoder' in name)):
                    net_para1.append(para)
            
            print('param: ', len(net_para1))

            for i in range(1,n):
                alpha = details.loc[i, 'alpha']
                if alpha == -1:
                    print("alpha is -1: ",net_para1.pop(i-1))
                else:
                    alpha_all.append(alpha)
            sorted_index = np.argsort(alpha_all)
            n = len(alpha_all)
            print('alpha', len(alpha_all))
            print('param: ', len(net_para1))
            if args.average:
                if args.average == 'softmax':
                    alpha_all = np.exp(alpha_all)
                elif args.average == 'sqrt':
                    alpha_all = np.sqrt(alpha_all)
                elif args.average == 'cbrt':
                    alpha_all = np.cbrt(alpha_all)
                elif args.average == 'log2':
                    alpha_all = np.log2(alpha_all)
                elif args.average == 'log':
                    alpha_all = np.log(alpha_all)
                alpha_all_weighted = alpha_all/np.sum(alpha_all)
                for i in range(0,n):
                    para_copy.append({'params': net_para1[i],
                                        'lr': get_custom_lr(epoch,WARMUP_STEPS,args.nemb) * n * alpha_all_weighted[i],})
            
            else:
                print('alpha_all: ', len(alpha_all))
                print('net_para: ', len(net_para1))
                n = len(alpha_all)
                # for i in range(n // 2):
                #     para_copy.append({
                #         'params':net_para1[sorted_index[i]],
                #         'lr': (0.35 * args.lr + 0.15 * args.lr * 2 * i / n) * (
                #                       1 + math.cos((epoch + 1) * math.pi / args.epoch_num)),
                #     })
                    
                # for i in range(n // 2, n, 1):
                #     para_copy.append({
                #         'params': net_para1[sorted_index[i]],
                #         'lr':(0.5 * args.lr + (i - n / 2) * (0.15 * args.lr * 2 / n)) * (
                #                       1 + math.cos((epoch + 1) * math.pi / args.epoch_num)),
                #     })
                for i in range(n // 2):
                    para_copy.append({
                        'params':net_para1[sorted_index[i]],
                        'lr': (1 - args.percentage) * get_custom_lr(epoch,WARMUP_STEPS,args.nemb),
                    })
                    
                for i in range(n // 2, n, 1):
                    para_copy.append({
                        'params': net_para1[sorted_index[i]],
                        'lr':(1 + args.percentage) * get_custom_lr(epoch,WARMUP_STEPS,args.nemb),
                    })
            
            para_save = [value for index, value in sorted(list(zip(np.argsort(alpha_all), para_copy)), key=itemgetter(0))]
            params_id = list(map(id, net_para1))
            other_params = list(filter(lambda p: id(p) not in params_id, model.parameters()))
            para_save.append({'params': other_params})  # save other optimizer info.
            optimizer = optim.Adam(para_save, betas=(0.9, 0.98), eps=1e-9, lr=get_custom_lr(epoch,WARMUP_STEPS,args.nemb))

            if args.record_lr:
                writer.add_scalars(main_tag='lr/layer',
                                tag_scalar_dict={str(i): optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))},
                                global_step=epoch)
            
                