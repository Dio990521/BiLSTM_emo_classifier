import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
import torch
import pickle
import numpy as np
from model.seq2seq_att import Seq2SeqAttentionSharedEmbedding
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
from utils.dataloader import Seq2SeqDataLoader
import argparse
from sklearn.model_selection import ShuffleSplit
from utils.evaluation import bleu
import utils.nn_utils as nn_utils
from utils.file_logger import get_file_logger
from utils.tokenizer import GloveTokenizer
from trainer_lstm_binary_test import *

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--batch_size', default=128, type=int, help="batch size")
parser.add_argument('--dim', default=700, type=int)
parser.add_argument('--att_dropout', default=0.1, type=float)
parser.add_argument('--no_tqdm', action='store_true')
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--save_models', action='store_true')
parser.add_argument('--mode', default='train', type=str, choices=['train', 'evaluate', 'interact'])
parser.add_argument('--load_model_path', default=None, type=str)
parser.add_argument('--input_feed', action='store_true')
parser.add_argument('--glorot_init', action='store_true')
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--model_path', type=str, default='checkpoint')
args = parser.parse_args()

if args.log_path is not None:
    dir_path = os.path.dirname(args.log_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
logger = get_file_logger(args.log_path)

pad_len = 20
batch_size = args.batch_size
emb_dim = 300
dim = args.dim
split_ratio = 0.1

data_path = 'data/nlpcc'
with open(os.path.join(data_path, 'word2id.bin'), 'br') as f:
    word2id = pickle.load(f)
with open(os.path.join(data_path, 'id2word.bin'), 'br') as f:
    id2word = pickle.load(f)

with open(os.path.join(data_path, 'nlpcc_train.bin'), 'br') as f:
    X, _, Y, _ = pickle.load(f)

vocab_size = len(word2id)

# X = X[:10000]
# Y = Y[:10000]

sss = ShuffleSplit(n_splits=1, test_size=split_ratio, random_state=0)
train_index, dev_index = next(sss.split(X))

X_train = [X[i] for i in train_index]
y_train = [Y[i] for i in train_index]
X_dev = [X[i] for i in dev_index]
y_dev = [Y[i] for i in dev_index]
training_set = Seq2SeqDataLoader(X_train, y_train, pad_len, word2id, glove_tokenizer=glove_tokenizer)
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
dev_set = Seq2SeqDataLoader(X_dev, y_dev, pad_len, word2id, glove_tokenizer=glove_tokenizer)
dev_loader = DataLoader(dev_set, batch_size=batch_size*3, shuffle=False)

GLOVE_EMB_PATH = 'sgns.weibo.char'
glove_tokenizer = GloveTokenizer(pad_len)

glove_tokenizer.build_embedding(GLOVE_EMB_PATH, dataset_name='nlpcc', voc_size=vocab_size, word2id=word2id, id2word=id2word)
print('vocab: ',glove_tokenizer.get_vocab_size())
# # Overfitting test

# vocab_size = 20000
# import random
# y_test = list(range(4, vocab_size))
# random.shuffle(y_test)
# y_test = y_test[:pad_len]
# X = [[5] * 20, [4] * 20]
# Y = [y_test, list(range(9, 16))]


# X_train = X[:4]
# y_train = Y[:4]
# X_dev = X_train
# y_dev = y_train
# training_set = Seq2SeqDataLoader(X_train, y_train, pad_len, word2id)
# train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
# dev_set = Seq2SeqDataLoader(X_dev, y_dev, pad_len, word2id)
# dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False)

y_dev_token = [[' '.join([id2word[idx] for idx in y][:pad_len]) for y in y_dev]]
model = Seq2SeqAttentionSharedEmbedding(
        emb_dim=emb_dim,
        vocab_size=glove_tokenizer.get_vocab_size(),
        src_hidden_dim=dim,
        trg_hidden_dim=dim,
        ctx_hidden_dim=dim,
        attention_mode='dot',
        batch_size=batch_size,
        bidirectional=False,
        pad_token_src=word2id['<pad>'],
        pad_token_trg=word2id['<pad>'],
        nlayers=1,
        nlayers_trg=2,
        dropout=0.2,
        att_dropout=args.att_dropout,
        word2id=word2id,
        max_decode_len=pad_len,
        id2word=id2word,
        input_feed=args.input_feed
    )

if args.glorot_init:
    nn_utils.glorot_init(model.parameters())

if args.load_model_path is not None:
    with open(args.load_model_path, 'br') as f:
        model.load_state_dict(torch.load(f))



def train():
    
    weight_mask = torch.ones(vocab_size).cuda()
    weight_mask[word2id['<pad>']] = torch.tensor(0).cuda()
    loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)  #
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1000):
        logger(f'Training on epoch={epoch} -------------------------')
        train_loss_sum = 0
        model.train()
        for i, (src, src_len, trg) in tqdm(enumerate(train_loader),
                                           total=int(len(training_set)/batch_size), disable=args.no_tqdm):
            optimizer.zero_grad()
            decoder_logit = model(src.cuda(), src_len.cuda(), trg.cuda())
            # trg_shift = torch.cat((torch.index_select(trg, 1, torch.LongTensor(list(range(1, pad_len)))),
            #                       torch.LongTensor(np.ones([trg.shape[0], 1])*word2id['<pad>'])), dim=1)
            trg_shift = trg[:, 1:]
            decoder_logit = decoder_logit[:, :-1, :]
            loss = loss_criterion(
                decoder_logit.reshape(-1, vocab_size),
                trg_shift.reshape(-1).cuda()
            )
            train_loss_sum += loss.item()*src.size()[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # del loss, decoder_logit
        logger(f"[Epoch {epoch}]: training Loss", train_loss_sum / len(training_set))

        evaluate()
        if True:
            torch.save(
                model.state_dict(),
                open(args.model_path + f'_epoch_{epoch}.bin', 'wb')
            )


def evaluate():
    model.eval()
    with torch.no_grad():
        gold_list = []
        pred_list = []
        for i, (src_test, src_len_test, trg_test)\
                in tqdm(enumerate(dev_loader), total=int(len(dev_set) / batch_size), disable=args.no_tqdm):
            batched_ouput = model.greedy_decode_batch(src_test.cuda(), src_len_test.cuda())
            gold_list.append(trg_test)
            batched_output_transpose = [[batched_ouput[step_output][batch_id]
                                        for step_output in range(len(batched_ouput))]
                                       for batch_id in range(len(batched_ouput[0]))]
            # trim from </s>
            trimed_output = [item[:item.index('</s>')] if '</s>' in item else item for item in batched_output_transpose]
            pred_list.extend(trimed_output)
        logger("BLUE score:", bleu(y_dev_token, pred_list))

if __name__ == '__main__':
    
    model.load_encoder_embedding(glove_tokenizer.get_embeddings())
    model.cuda()

    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate()
    else:
        raise NotImplemented("Mode not implemented")
