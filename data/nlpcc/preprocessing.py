"""
Preprocess NLPCC dataset, by Chenyang in Sept, 2020
"""
import json
import pickle

f_path = "train_data.json"
train_data = json.load(open(f_path, 'r'))

# 1. count words
wordcount = {}
for utter in train_data:
    _X, _Y = utter
    x_sent, x_emo = _X
    y_sent, y_emo = _Y
    for word in (x_sent.strip() + ' ' + y_sent.strip()).split():
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
print("Num of total words:", len(wordcount))

# 2. truncate words
MIN_WORD_COUNT = 20
after_filtering = dict(filter(lambda x: x[1] >= MIN_WORD_COUNT, wordcount.items()))
print(f"{len(after_filtering)} of words occurred more than {MIN_WORD_COUNT} times")

# 3. Build word2id
word2id = {
    '<unk>': 0,
    '<pad>': 1,
    '<s>': 2,
    '</s>': 3
}
idx = len(word2id)
for word, _ in after_filtering.items():
    word2id[word] = idx
    idx += 1

with open('word2id.bin', 'wb') as f:
    pickle.dump(word2id, f)

# 3.1 get id2word
with open('id2word.bin', 'wb') as f:
    id2word = {x[1]: x[0] for x in word2id.items()}
    pickle.dump(id2word, f)

assert len(id2word) == len(word2id)

# 4. convert textual data to index
x_emo_list = []
y_emo_list = []
x_ids_list = []
y_ids_list = []
for utter in train_data:
    _X, _Y = utter
    # convert X
    x_sent, x_emo = _X
    x_ids_list.append([word2id[word] if word in word2id else word2id['<unk>']
                       for word in x_sent.strip().split()])
    x_emo_list.append(int(x_emo))

    # convert Y
    y_sent, y_emo = _Y
    y_ids_list.append([word2id[word] if word in word2id else word2id['<unk>']
                       for word in y_sent.strip().split()])
    y_emo_list.append(int(y_emo))

with open('nlpcc_train.bin', 'bw') as f:
    pickle.dump((x_ids_list, x_emo_list, y_ids_list, y_emo_list), f)

