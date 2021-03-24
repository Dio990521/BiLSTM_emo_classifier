import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.early_stopping import EarlyStopping
import numpy as np
import pickle as pkl
import copy
from tqdm import tqdm
from sklearn.metrics import *
import argparse
from copy import deepcopy
from utils.seq2emo_metric import get_metrics, get_multi_metrics, jaccard_score, report_all, get_single_metrics
from model.binary_roberta import BinaryBertClassifier
from transformers import BertTokenizer, AdamW, BertConfig
from utils.scheduler import get_cosine_schedule_with_warmup
from utils.others import find_majority
import random
from utils import nn_utils
from utils.file_logger import get_file_logger


parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--batch_size', default=16, type=int, help="batch size")
parser.add_argument('--postname', default='', type=str, help="post name")
parser.add_argument('--gamma', default=0.2, type=float, help="post name")
parser.add_argument('--loss', default='ce', type=str, help="loss function ce/focal")
parser.add_argument('--dataset', default='sem18', type=str, choices=['sem18', 'goemotions', 'bmet'])
parser.add_argument('--criterion', default='jaccard', type=str,
                    help='criterion to prevent overfitting, currently support f1 and loss')
parser.add_argument('--bert', default='base', type=str, help="bert size [base/large]")
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--warmup_epoch', default=2, type=int, help='')
parser.add_argument('--stop_epoch', default=10, type=int, help='')
parser.add_argument('--max_epoch', default=20, type=int, help='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--en_lr', type=float, default=5e-5)
parser.add_argument('--de_lr', default=5e-5, type=float, help="decoder learning rate")
parser.add_argument('--attention', default='dot', type=str, help='general/mlp/dot')
parser.add_argument('--de_dim', default=400, type=int, help="dimension")
parser.add_argument('--decoder_dropout', default=0, type=float, help='dropout rate')
parser.add_argument('--attention_dropout', default=0.25, type=float, help='dropout rate')
parser.add_argument('--input_feeding', action='store_true')
parser.add_argument('--glorot_init', action='store_true')
parser.add_argument('--huang_init', action='store_true')
parser.add_argument('--normal_init', action='store_true')
parser.add_argument('--unify_decoder', action='store_true')
parser.add_argument('--eval_every', type=int, default=500)
parser.add_argument('--patience', default=3, type=int, help='dropout rate')
parser.add_argument('--min_lr_ratio', default=0.1, type=float, help='')
parser.add_argument('--en_de_activate_function', default='tanh', type=str)
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)


args = parser.parse_args()

if args.log_path is not None:
    dir_path = os.path.dirname(args.log_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

logger = get_file_logger(args.log_path)  # Note: this is ugly, but I am lazy


# if args.bert == 'base':
#     BERT_MODEL = 'bert-base-chinese'
#     SRC_HIDDEN_DIM = 768
# else:
#     raise ValueError('Specified BERT model NOT supported!!')


# 768 for roberta_l12 1024 for roberta_large
folder_path = 'roberta_large'
vocab_path = f'{folder_path}/vocab.txt'
config_path = f'{folder_path}/config.json'
model_path = f'{folder_path}/pytorch_model.bin'
SRC_HIDDEN_DIM = 1024  # 768 for roberta_l12 1024 for roberta_large

tokenizer = BertTokenizer.from_pretrained(vocab_path)
config = BertConfig.from_pretrained(config_path)


NUM_FOLD = 5
ENCODER_LEARNING_RATE = args.en_lr
PAD_LEN = 50
MIN_LEN_DATA = 3
BATCH_SIZE = args.batch_size
TGT_HIDDEN_DIM = args.de_dim
ATTENTION = args.attention
PATIENCE = args.patience
WARMUP_EPOCH = args.warmup_epoch
STOP_EPOCH = args.stop_epoch

# Get BERT tokenizer

# BERT optimizer setup
max_grad_norm = 1.0
# num_training_steps = 1000
# num_warmup_steps = 100
# warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1
# fix random seeds
RANDOM_SEED = args.seed
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

data_path_postfix = ''
if args.dataset in ['sem18', 'goemotions']:
    LOAD_TEST_SPLIT = True
    data_path_postfix = '_split'
else:
    LOAD_TEST_SPLIT = False

data_pkl_path = 'data/' + args.dataset + data_path_postfix + '_data.pkl'

EMOS = ['anger', 'disgust', 'happiness', 'like', 'sadness', 'none']


def get_emotion(file, EMOS, EMOS_DIC):
    text_list = []
    label_list = []
    with open(file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            text, emotion = line.split('\t')
            text_list.append(text)
            label_list.append(int(EMOS_DIC[emotion.rstrip("\n")]))
    return text_list, label_list


def load_NLPCC_data():
    EMOS_DIC = {}
    for idx, emo in enumerate(EMOS):
        EMOS_DIC[emo] = idx
    file1 = 'data/nlpcc/nlpcc2013_2014_adjust.txt'

    X, y = get_emotion(file1, EMOS, EMOS_DIC)
    X_train = X[:18000]
    y_train = y[:18000]
    X_dev = X[18000:19000]
    y_dev = y[18000:19000]
    X_test = X[19000:]
    y_test = y[19000:]
    X_train_dev = X_train + X_dev
    y_train_dev = y_train + y_dev
    # preprocess

    return X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, 'nlpcc'


NUM_EMO = len(EMOS)

X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = \
    load_NLPCC_data()


class TestDataReader(Dataset):
    def __init__(self, X, pad_len, max_size=None):
        self.glove_ids_len = []
        self.pad_len = pad_len
        self.tokens = []
        self.token_masks = []
        self.pad_int = 0
        self.__read_data(X)

    def __read_data(self, data_list):
        for X in data_list:

            X = tokenizer.tokenize(X)
            X = ['[CLS]'] + X + ['[SEP]']
            X = tokenizer.convert_tokens_to_ids(X)
            X_len = len(X)

            if len(X) > self.pad_len:
                X = X[:self.pad_len]
                mask = [1] * self.pad_len
            else:
                X = X + [self.pad_int] * (self.pad_len - len(X))
                mask = [1] * X_len + [0] * (self.pad_len - X_len)

            self.tokens.append(X)
            self.token_masks.append(mask)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return torch.LongTensor(self.tokens[idx]), \
               torch.LongTensor(self.token_masks[idx])


class TrainDataReader(TestDataReader):
    def __init__(self, X, y, pad_len, max_size=None):
        super(TrainDataReader, self).__init__(X, pad_len, max_size)
        self.y = []
        self.__read_target(y)

    def __read_target(self, y):
        self.y = y

    def __getitem__(self, idx):
        return torch.LongTensor(self.tokens[idx]), \
               torch.LongTensor(self.token_masks[idx]), \
               torch.LongTensor([self.y[idx]])


def eval(model, best_model, loss_criterion, es, dev_loader, dev_data):
    pred_list = []
    gold_list = []
    test_loss_sum = 0
    exit_training = False
    model.eval()
    for _, (_data, _mask, _label) in enumerate(dev_loader):
        with torch.no_grad():
            decoder_logit = model(_data.cuda(), _mask.cuda())
            test_loss = loss_criterion(
                decoder_logit.view(-1, decoder_logit.shape[-1]),
                _label.view(-1).cuda()
            )
            test_loss_sum += test_loss.data.cpu().numpy() * _data.shape[0]
            gold_list.append(_label.numpy())

            pred_list.append(np.argmax(decoder_logit.data.cpu().numpy(), axis=-1))
            del decoder_logit, test_loss
            # break

    preds = np.concatenate(pred_list, axis=0)
    gold = np.concatenate(gold_list, axis=0)
    metric = get_metrics(gold, preds)
    # report_all(gold_list, pred_list)
    jaccard = jaccard_score(gold, preds)
    logger("Evaluation results:")
    # show_classification_report(binary_gold, binary_preds)
    logger("Evaluation Loss", test_loss_sum / len(dev_data))

    logger('Normal: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4], 'micro P', metric[5],
          'micro R', metric[6])
    metric_2 = get_multi_metrics(gold, preds)
    logger('Multi only: h_loss:', metric_2[0], 'macro F', metric_2[1], 'micro F', metric_2[4])
    logger('Jaccard:', jaccard)

    if args.criterion == 'loss':
        criterion = test_loss_sum
    elif args.criterion == 'macro':
        criterion = 1 - metric[1]
    elif args.criterion == 'micro':
        criterion = 1 - metric[4]
    elif args.criterion == 'h_loss':
        criterion = metric[0]
    elif args.criterion == 'jaccard':
        criterion = 1 - jaccard
    else:
        raise ValueError

    if es.step(criterion):  # overfitting
        del model
        logger('overfitting, loading best model ...')
        model = best_model
        exit_training = True
    else:
        if es.is_best():
            if best_model is not None:
                del best_model
            logger('saving best model ...')
            best_model = deepcopy(model)
        else:
            logger(f'patience {es.cur_patience} not best model , ignoring ...')
            if best_model is None:
                best_model = deepcopy(model)

    return model, best_model, exit_training


def train(X_train, y_train, X_dev, y_dev, X_test, y_test):
    train_data = TrainDataReader(X_train, y_train, PAD_LEN)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    dev_data = TrainDataReader(X_dev, y_dev, PAD_LEN)
    dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False)

    test_data = TrainDataReader(X_test, y_test, PAD_LEN)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


    is_broken = False

    model = BinaryBertClassifier(
        hidden_dim=SRC_HIDDEN_DIM,
        num_label=NUM_EMO,
        args=args
    )
    model.init_encoder(model_path, config_path)
    model.cuda()

    loss_criterion = nn.CrossEntropyLoss()  #

    # Encoder setup
    learning_rate, adam_epsilon, weight_decay, warmup_steps = ENCODER_LEARNING_RATE, 1e-8, 0, 0
    no_decay = ['bias', 'LayerNorm.weight']
    encoder_optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and n.startswith('encoder')],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and n.startswith('encoder')],
         'weight_decay': 0.0}
    ]
    encoder_optimizer = AdamW(encoder_optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    # Decoder setup
    decoder_optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if n.startswith("decoder")], 'lr': args.de_lr}]
    decoder_optimizer = optim.Adam(decoder_optimizer_grouped_parameters)

    if args.glorot_init:
        logger('use glorot initialization')
        for group in decoder_optimizer_grouped_parameters:
            nn_utils.glorot_init(group['params'])

    if args.huang_init:
        nn_utils.huang_init(model.named_parameters(), uniform=not args.normal_init, startswith='decoder')

    if args.scheduler:
        epoch_to_step = int(len(train_data) / BATCH_SIZE)
        encoder_scheduler = get_cosine_schedule_with_warmup(
            encoder_optimizer, num_warmup_steps=WARMUP_EPOCH * epoch_to_step,
            num_training_steps=STOP_EPOCH * epoch_to_step,
            min_lr_ratio=args.min_lr_ratio
        )
        decoder_scheduler = get_cosine_schedule_with_warmup(
            encoder_optimizer, num_warmup_steps=0,  # NOTE: decoder start steps set to 0, hardcoded warning
            num_training_steps=STOP_EPOCH * epoch_to_step,
            min_lr_ratio=args.min_lr_ratio
        )

    es = EarlyStopping(patience=PATIENCE)
    best_model = None
    exit_training = None
    EVAL_EVERY = int(len(train_data) / BATCH_SIZE / 4)

    update_step = 0
    for epoch in range(1, args.max_epoch):
        logger('Epoch: ' + str(epoch) + '===================================')
        train_loss = 0

        for i, (src, mask, label) in tqdm(enumerate(train_loader),
                                          total=len(train_data) / BATCH_SIZE):
            model.train()
            update_step += 1
            if args.scheduler:
                encoder_scheduler.step()
                decoder_scheduler.step()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            decoder_logit = model(src.cuda(), mask.cuda())

            loss = loss_criterion(
                decoder_logit.view(-1, decoder_logit.shape[-1]),
                label.view(-1).cuda()
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            encoder_optimizer.step()
            decoder_optimizer.step()
            # scheduler.step()
            train_loss += loss.data.cpu().numpy() * src.shape[0]
            del decoder_logit, loss
            # break
            if update_step % EVAL_EVERY == 0 and args.eval_every is not None:
                model, best_model, exit_training = eval(model, best_model, loss_criterion, es, dev_loader, dev_data)
                if exit_training:
                    break

        logger(f"Training Loss for epoch {epoch}:", train_loss / len(train_data))
        model, best_model, exit_training = eval(model, best_model, loss_criterion, es, dev_loader, dev_data)
        if exit_training:
            break


    pred_list = []
    gold_list = []
    model.eval()
    for _, (_data, _mask, _label) in enumerate(test_loader):
        with torch.no_grad():
            decoder_logit = model(_data.cuda(), _mask.cuda())
            pred_list.append(np.argmax(decoder_logit.data.cpu().numpy(), axis=-1))

            gold_list.append(_label.numpy())
            del decoder_logit
        # break

    torch.save(model, 'nlpcc_roberta_large.pt')
    # pred_list_2 = np.concatenate(pred_list, axis=0)[:, 1]
    preds = np.concatenate(pred_list, axis=0)
    gold = np.concatenate(gold_list, axis=0)

    binary_gold = gold
    binary_preds = preds
    logger("NOTE, this is on the test set")
    metric = get_metrics(binary_gold, binary_preds)
    logger('Normal: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
    metric = get_multi_metrics(binary_gold, binary_preds)
    logger('Multi only: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
    # show_classification_report(binary_gold, binary_preds)
    logger('Jaccard:', jaccard_score(gold, preds))

    return binary_gold, binary_preds


def show_classification_report(gold, pred):
    from sklearn.metrics import classification_report
    logger(classification_report(gold, pred, target_names=EMOS, digits=4))


def main():
    if not LOAD_TEST_SPLIT:
        global X, y
    else:
        global X_train_dev, X_test, y_train_dev, y_test

    from sklearn.model_selection import ShuffleSplit, KFold
    if not LOAD_TEST_SPLIT:
        ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        ss.get_n_splits(X, y)
        train_index, test_index = next(ss.split(y))
        X_train_dev, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train_dev, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    kf = KFold(n_splits=NUM_FOLD, random_state=0)

    gold_list = None
    # all_preds = []
    for i, (train_index, dev_index) in enumerate(kf.split(y_train_dev)):
        logger('STARTING Fold -----------', i + 1)
        X_train, X_dev = [X_train_dev[i] for i in train_index], [X_train_dev[i] for i in dev_index]
        y_train, y_dev = [y_train_dev[i] for i in train_index], [y_train_dev[i] for i in dev_index]

        gold_list, pred_list = train(X_train, y_train, X_dev, y_dev, X_test, y_test)
        # all_preds.append(pred_list)
        break

    # all_preds = np.stack(all_preds, axis=0)

    # shape = all_preds[0].shape
    # mj = np.zeros(shape)
    # for m in range(shape[0]):
    #     for n in range(shape[1]):
    #         mj[m, n] = find_majority(np.asarray(all_preds[:, m, n]).reshape((-1)))[0]
    final_pred = pred_list

    logger('Final test by majority voting:')
    show_classification_report(gold_list, final_pred)
    metric = get_metrics(gold_list, final_pred)
    logger('Normal: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
    metric = get_multi_metrics(gold_list, final_pred)
    logger('Multi only: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
    metric = get_single_metrics(gold_list, final_pred)
    logger('Single only: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
    logger('Jaccard:', jaccard_score(gold_list, final_pred))
    logger('Bert Binary', args)

    if args.output_path is not None:
        with open(args.output_path, 'bw') as _f:
            pkl.dump(final_pred, _f)
main()
