import sys
import os
import numpy as np
import argparse
import torch
import time

object_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(object_path)

from tasks.task import *
from util.data_util import *
from models.model import *


def parse_args():
    config_args = {
        'lr': 0.0005,
        'dropout_gat': 0.3,
        'dropout': 0.3,
        'cuda': 0,
        'epochs_gat': 3000,
        'epochs': 1000,
        'weight_decay_gat': 1e-5,
        'weight_decay': 0,
        'seed': 10010,
        'model': 'IMF',
        'num-layers': 3,
        'dim': 256,
        'r_dim': 256,
        'k_w': 10,
        'k_h': 20,
        'n_heads': 2,
        'dataset': 'FB15K-237',
        'pre_trained': 0,
        'encoder': 0,
        'image_features': 1,
        'text_features': 1,
        'patience': 5,
        'eval_freq': 10,
        'lr_reduce_freq': 500,
        'gamma': 1.0,
        'bias': 1,
        'neg_num': 2,
        'neg_num_gat': 2,
        'alpha': 0.2,
        'alpha_gat': 0.2,
        'out_channels': 32,
        'kernel_size': 3,
        'batch_size': 256,
        'save': 1
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", action="append", default=val)
    args = parser.parse_args()
    return args

args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
print(f'Using: {args.device}')
torch.cuda.set_device(args.cuda)
for k, v in list(vars(args).items()):
    print(str(k) + ':' + str(v))

entity2id, relation2id, img_features, text_features, train_data, val_data, test_data = load_data(args.dataset)
print("Training data {:04d}".format(len(train_data[0])))

corpus = ConvKBCorpus(args, train_data, val_data, test_data, entity2id, relation2id)

args.entity2id = entity2id
args.relation2id = relation2id


def train_encoder(args):
    model = GAT(args)
    LOG_TRAIN.info(str(model))
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=float(args.gamma))
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    LOG_TRAIN.info(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        model = model.to(args.device)

    # Train Model
    t_total = time.time()
    # corpus.batch_size = len(corpus.train_triples)
    corpus.neg_num = 2

    for epoch in range(args.epochs_gat):
        model.train()
        t = time.time()
        np.random.shuffle(corpus.train_triples)
        # batch
        for i in range(corpus.max_batch_num):
            train_indices, train_values = corpus.get_batch(i)
            train_indices = torch.LongTensor(train_indices)
            if args.cuda is not None and int(args.cuda) >= 0:
                train_indices = train_indices.to(args.device)

            optimizer.zero_grad()
            entity_embed, relation_embed = model.forward(corpus.train_adj_matrix, train_indices)
            loss = model.loss_func(train_indices, entity_embed, relation_embed)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        LOG_TRAIN.info("Epoch {} , epoch_time {}".format(epoch, time.time() - t))
        if args.save:
            # torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/GAT_{epoch}.pth')
            pass

    LOG_TRAIN.info("GAT training finished! Total time is {}".format(time.time()-t_total))


if __name__ == '__main__':
    # task = SolrInitTask()
    # task.run_task()
    train_encoder(args)
