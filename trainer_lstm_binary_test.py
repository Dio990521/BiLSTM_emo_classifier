import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from model.binary_lstm import BinaryLSTMClassifier
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
import pickle as pkl
from utils.seq2emo_metric import get_metrics, get_multi_metrics, jaccard_score, report_all, get_single_metrics
from utils.tokenizer import GloveTokenizer
from copy import deepcopy
from allennlp.modules.elmo import Elmo, batch_to_ids
import argparse
from data.data_loader import load_BMET_data, load_cbet_data, load_sem18_data, load_goemotions_data
from utils.scheduler import get_cosine_schedule_with_warmup
import utils.nn_utils as nn_utils
from utils.others import find_majority
from utils.file_logger import get_file_logger

torch.backends.cudnn.enabled = False
# Argument parser
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--batch_size', default=32, type=int, help="batch size")
parser.add_argument('--pad_len', default=50, type=int, help="batch size")
parser.add_argument('--postname', default='', type=str, help="post name")
parser.add_argument('--gamma', default=0.2, type=float, help="post name")
parser.add_argument('--folds', default=5, type=int, help="num of folds")
parser.add_argument('--en_lr', default=5e-4, type=float, help="encoder learning rate")
parser.add_argument('--de_lr', default=5e-4, type=float, help="decoder learning rate")
parser.add_argument('--loss', default='ce', type=str, help="loss function ce/focal")
parser.add_argument('--dataset', default='nlpcc', type=str, choices=['sem18', 'goemotions', 'bmet', 'nlpcc'])
parser.add_argument('--en_dim', default=800, type=int, help="dimension")
parser.add_argument('--de_dim', default=400, type=int, help="dimension")
parser.add_argument('--criterion', default='micro', type=str, choices=['jaccard', 'macro', 'micro', 'h_loss'])
parser.add_argument('--glove_path', default='data/glove.840B.300d.txt', type=str)
parser.add_argument('--attention', default='bert', type=str, choices=['bert', 'attentive', 'None'])
parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
parser.add_argument('--encoder_dropout', default=0, type=float, help='dropout rate')
parser.add_argument('--decoder_dropout', default=0, type=float, help='dropout rate')
parser.add_argument('--attention_dropout', default=0.25, type=float, help='dropout rate')
parser.add_argument('--patience', default=10, type=int, help='dropout rate')
parser.add_argument('--download_elmo', action='store_true')
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--glorot_init', action='store_true')
parser.add_argument('--warmup_epoch', default=0, type=int, help='')
parser.add_argument('--stop_epoch', default=50, type=int, help='')
parser.add_argument('--max_epoch', default=100, type=int, help='')
parser.add_argument('--min_lr_ratio', default=0.1, type=float, help='')
parser.add_argument('--fix_emb', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--input_feeding', action='store_true')
parser.add_argument('--dev_split_seed', type=int, default=0)
parser.add_argument('--huang_init', action='store_true')
parser.add_argument('--normal_init', action='store_true')
parser.add_argument('--unify_decoder', action='store_true')
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--no_cross', action='store_true')
args = parser.parse_args(args=[])

if args.log_path is not None:
    dir_path = os.path.dirname(args.log_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

logger = get_file_logger(args.log_path)  # Note: this is ugly, but I am lazy

SRC_EMB_DIM = 300
MAX_LEN_DATA = args.pad_len
PAD_LEN = MAX_LEN_DATA
MIN_LEN_DATA = 3
BATCH_SIZE = args.batch_size
CLIPS = 0.666
GAMMA = 0.5
SRC_HIDDEN_DIM = args.en_dim
TGT_HIDDEN_DIM = args.de_dim
VOCAB_SIZE = 60000
ENCODER_LEARNING_RATE = args.en_lr
DECODER_LEARNING_RATE = args.de_lr
ATTENTION = args.attention
PATIENCE = args.patience
WARMUP_EPOCH = args.warmup_epoch
STOP_EPOCH = args.stop_epoch
MAX_EPOCH = args.max_epoch
RANDOM_SEED = args.seed

# Seed
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Init Elmo model

GLOVE_EMB_PATH = 'sgns.weibo.char'
glove_tokenizer = GloveTokenizer(PAD_LEN)

data_path_postfix = ''
if args.dataset in ['sem18', 'goemotions', 'bmet', 'nlpcc']:
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
        self.glove_ids = []
        self.glove_ids_len = []
        self.pad_len = pad_len
        self.build_glove_ids(X)

    def build_glove_ids(self, X):
        for src in X:
            glove_id, glove_id_len = glove_tokenizer.encode_ids_pad(src)
            self.glove_ids.append(glove_id)
            self.glove_ids_len.append(glove_id_len)

    def __len__(self):
        return len(self.glove_ids)

    def __getitem__(self, idx):
        return torch.LongTensor(self.glove_ids[idx]), \
               torch.LongTensor([self.glove_ids_len[idx]])


class TrainDataReader(TestDataReader):
    def __init__(self, X, y, pad_len, max_size=None):
        super(TrainDataReader, self).__init__(X, pad_len, max_size)
        self.y = []
        self.read_target(y)

    def read_target(self, y):
        self.y = y

    def __getitem__(self, idx):
        return torch.LongTensor(self.glove_ids[idx]), \
               torch.LongTensor([self.glove_ids_len[idx]]), \
               torch.LongTensor([self.y[idx]])



def show_classification_report(gold, pred):
    from sklearn.metrics import classification_report
    logger(classification_report(gold, pred, target_names=EMOS, digits=4, labels=list(range(NUM_EMO))))

def eval(model, best_model, loss_criterion, es, dev_loader, dev_set, y_dev):
    # Evaluate
    exit_training = False
    model.eval()
    test_loss_sum = 0
    preds = []
    # gold = []
    logger("Evaluating:")
    for i, (src, src_len, trg) in tqdm(enumerate(dev_loader), total=int(len(dev_set) / BATCH_SIZE), disable=True):
        with torch.no_grad():
            
            decoder_logit = model(src.cuda(), src_len.cuda())

            test_loss = loss_criterion(
                decoder_logit,
                trg.view(-1).cuda()
            )
            test_loss_sum += test_loss.data.cpu().numpy() * src.shape[0]
            # gold.append(trg.data.numpy())
            preds.append(np.argmax(decoder_logit.data.cpu().numpy(), axis=-1))
            del decoder_logit

    preds = np.concatenate(preds, axis=0)
    gold = np.asarray(y_dev)

    print('EVAL---------: ')
    show_classification_report(gold, preds)

    metric = get_metrics(gold, preds)
    logger("Evaluation results:")
    # show_classification_report(binary_gold, binary_preds)
    logger("Evaluation Loss", test_loss_sum / len(dev_set))

    if args.criterion == 'loss':
        criterion = test_loss_sum
    elif args.criterion == 'macro':
        criterion = 1 - metric[1]
    elif args.criterion == 'micro':
        criterion = 1 - metric[4]
    elif args.criterion == 'h_loss':
        criterion = metric[0]

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
    train_set = TrainDataReader(X_train, y_train, MAX_LEN_DATA)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    dev_set = TrainDataReader(X_dev, y_dev, MAX_LEN_DATA)
    dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE * 3, shuffle=False)

    test_set = TestDataReader(X_test, MAX_LEN_DATA)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE * 3, shuffle=False)

    # Model initialize
    model = BinaryLSTMClassifier(
        emb_dim=SRC_EMB_DIM,
        vocab_size=glove_tokenizer.get_vocab_size(),
        num_label=NUM_EMO,
        hidden_dim=SRC_HIDDEN_DIM,
        attention_mode=ATTENTION,
        args=args
    )

    if args.fix_emb:
        para_group = [
            {'params': [p for n, p in model.named_parameters() if n.startswith("encoder") and
                        not 'encoder.embeddings' in n], 'lr': args.en_lr},
            {'params': [p for n, p in model.named_parameters() if n.startswith("decoder")], 'lr': args.de_lr}]
    else:
        para_group = [
            {'params': [p for n, p in model.named_parameters() if n.startswith("encoder")], 'lr': args.en_lr},
            {'params': [p for n, p in model.named_parameters() if n.startswith("decoder")], 'lr': args.de_lr}]
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(para_group)
    if args.scheduler:
        epoch_to_step = int(len(train_set) / BATCH_SIZE)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_EPOCH * epoch_to_step,
            num_training_steps=STOP_EPOCH * epoch_to_step,
            min_lr_ratio=args.min_lr_ratio
        )

    if args.glorot_init:
        logger('use glorot initialization')
        for group in para_group:
            nn_utils.glorot_init(group['params'])

    if args.huang_init:
        nn_utils.huang_init(model.named_parameters(), uniform=not args.normal_init)
    model.load_encoder_embedding(glove_tokenizer.get_embeddings(), fix_emb=args.fix_emb)
    model.cuda()

    # Start training
    EVAL_EVERY = int(len(train_set) / BATCH_SIZE / 4)
    best_model = None
    es = EarlyStopping(patience=PATIENCE)
    update_step = 0
    exit_training = False

    for epoch in range(1, MAX_EPOCH + 1):
        train_pred = []
        train_gold_list = []
        logger('Training on epoch=%d -------------------------' % (epoch))
        train_loss_sum = 0
        # print('Current encoder learning rate', scheduler.get_lr())
        # print('Current decoder learning rate', scheduler.get_lr())
        for i, (src, src_len, trg) in tqdm(enumerate(train_loader), total=int(len(train_set) / BATCH_SIZE)):
            model.train()
            update_step += 1

            # print('i=%d: ' % (i))
            # trg = torch.index_select(trg, 1, torch.LongTensor(list(range(1, len(EMOS)+1))))
            if args.scheduler:
                scheduler.step()

            optimizer.zero_grad()
            
            decoder_logit = model(src.cuda(), src_len.cuda())

            train_pred.append(np.argmax(decoder_logit.data.cpu().numpy(), axis=-1))
            gold = np.asarray(trg)
            trg_index = []
            for i in range(gold.shape[0]):
                train_gold_list.append(gold[i])
            loss = loss_criterion(decoder_logit, trg.view(-1).cuda())
            loss.backward()
            train_loss_sum += loss.data.cpu().numpy() * src.shape[0]

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPS)
            optimizer.step()

            if update_step % EVAL_EVERY == 0 and args.eval_every is not None:
                model, best_model, exit_training = eval(model, best_model, loss_criterion, es, dev_loader, dev_set,
                                                        y_dev)
                if exit_training:
                    break

        logger(f"Training Loss for epoch {epoch}:", train_loss_sum / len(train_set))
        if not train_pred == []:
            print('TRAIN---------: ')
            train_pred = np.concatenate(train_pred, axis=0)
            train_gold_list = np.array(train_gold_list)
            show_classification_report(train_gold_list, train_pred)
        # model, best_model, exit_training = eval(model, best_model, loss_criterion, es, dev_loader, dev_set)
        if exit_training:
            break

    # final_testing
    model.eval()
    preds = []
    logger("Testing:")
    for i, (src, src_len) in tqdm(enumerate(test_loader), total=int(len(test_set) / BATCH_SIZE)):
        with torch.no_grad():
            
            decoder_logit = model(src.cuda(), src_len.cuda())
            preds.append(np.argmax(decoder_logit.data.cpu().numpy(), axis=-1))
            del decoder_logit

    preds = np.concatenate(preds, axis=0)
    gold = np.asarray(y_test)
    #preds = np.argmax(preds, axis=-1)

    logger("NOTE, this is on the test set")
    #metric = get_metrics(gold, preds)
    #logger('Normal: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
    # metric = get_multi_metrics(binary_gold, binary_preds)
    # logger('Multi only: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
    # show_classification_report(binary_gold, binary_preds)
    # logger('Jaccard:', jaccard_score(gold, preds))
    return gold, preds, model

def inference_emotion(data_loader, data_set, batch_size):
    model = BinaryLSTMClassifier(
        emb_dim=SRC_EMB_DIM,
        vocab_size=34177,
        num_label=NUM_EMO,
        hidden_dim=SRC_HIDDEN_DIM,
        attention_mode=ATTENTION,
        args=args
    )
    model.cuda()
    all_preds = []
    for i in range(1,6):
        model.load_state_dict(torch.load('saved_model/emotion_classifier' + str(i) + '.pt'))
        model.eval()
        preds = []
        for i, (src, src_len) in tqdm(enumerate(data_loader), total=int(len(data_set) / batch_size)):
            with torch.no_grad():
            
                decoder_logit = model(src.cuda(), src_len.cuda())
                preds.append(np.argmax(decoder_logit.data.cpu().numpy(), axis=-1))
                del decoder_logit

        preds = np.concatenate(preds, axis=0)
        all_preds.append(preds)
        
    all_preds = np.stack(all_preds, axis=0)
    shape = all_preds[0].shape
    mj = np.zeros(shape[0])
    for m in range(shape[0]):
        mj[m] = find_majority(np.asarray(all_preds[:, m]).reshape((-1)))[0]

    final_pred = mj
    return final_pred

def main():
    if not LOAD_TEST_SPLIT:
        global X, y
        ALL_TRAINING = X
    else:
        global X_train_dev, X_test, y_train_dev, y_test
        ALL_TRAINING = X_train_dev + X_test
    glove_tokenizer.build_tokenizer(ALL_TRAINING, vocab_size=VOCAB_SIZE)
    glove_tokenizer.build_embedding(GLOVE_EMB_PATH, dataset_name=data_set_name)

    from sklearn.model_selection import ShuffleSplit, KFold

    if not LOAD_TEST_SPLIT:
        ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        ss.get_n_splits(X, y)
        train_index, test_index = next(ss.split(y))
        X_train_dev, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train_dev, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    kf = KFold(n_splits=args.folds, random_state=args.dev_split_seed)
    # kf.get_n_splits(X_train_dev)

    all_preds = []
    gold_list = None

    for i, (train_index, dev_index) in enumerate(kf.split(y_train_dev)):
        logger('STARTING Fold -----------', i + 1)
        X_train, X_dev = [X_train_dev[i] for i in train_index], [X_train_dev[i] for i in dev_index]
        y_train, y_dev = [y_train_dev[i] for i in train_index], [y_train_dev[i] for i in dev_index]

        gold_list, pred_list, model = train(X_train, y_train, X_dev, y_dev, X_test, y_test)
        all_preds.append(pred_list)
        #torch.save(model.state_dict(), 'saved_model/emotion_classifier' + str(i+1) + '.pt')
        #break
    all_preds = np.stack(all_preds, axis=0)
    shape = all_preds[0].shape
    mj = np.zeros(shape[0])
    for m in range(shape[0]):
        mj[m] = find_majority(np.asarray(all_preds[:, m]).reshape((-1)))[0]

    final_pred = mj

    print('TEST---------: ')
    show_classification_report(gold_list, final_pred)
    metric = get_metrics(gold_list, final_pred)
    logger('Normal: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
    # metric = get_multi_metrics(gold_list, final_pred)
    # logger('Multi only: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
    # metric = get_single_metrics(gold_list, final_pred)
    # logger('Single only: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])

    # logger('Final Jaccard:', jaccard_score(gold_list, final_pred))
    logger(os.path.basename(__file__))
    logger(args)


if __name__ == '__main__':
    main()
