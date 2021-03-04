import pandas as pd
import numpy as np
from nltk.corpus import stopwords


def cbet_data(file_path='data/CBET.csv', remove_stop_words=True, get_text=True, preprocess=True, multi=False, vector=False):
    NUM_CLASS = 9
    emo_list = ["anger", "fear", "joy", "love", "sadness", "surprise", "thankfulness", "disgust", "guilt"]
    stop_words = set(stopwords.words('english'))

    label = []
    train_text = []
    df = pd.read_csv(file_path)
    for i, row in df.iterrows():
        if get_text:
            from utils.tweet_processor import tweet_process
            text = row['text']
            if preprocess:
                text = tweet_process(text)
            if remove_stop_words:
                text = ' '.join([x for x in text.split() if x not in stop_words])
            train_text.append(text)

        emo_one_hot = row[emo_list]
        emo_one_hot = np.asarray(emo_one_hot)
        if not multi:
            if sum(emo_one_hot) != 1:
                continue
            emo_idx = np.argmax(emo_one_hot)
        else:
            if not vector:
                emo_idx = np.where(emo_one_hot == 1)[0].tolist()
            else:
                emo_idx = emo_one_hot
        label.append(emo_idx)

    return train_text, label, emo_list, NUM_CLASS


def isear_data(file_path='data/ISEAR.csv', remove_stop_words=True, get_text=True, preprocess=True):
    stop_words = set(stopwords.words('english'))
    NUM_CLASS = 7
    emo_list = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
    attributes = ['SIT']
    target = ['EMOT']
    loader = IsearLoader(attributes, target, True)
    data = loader.load_isear(file_path)
    train_text = []
    if get_text:
        text_all = data.get_freetext_content()  # returns attributes
        for text in text_all:
            from utils.tweet_processor import tweet_process
            if preprocess:
                text = tweet_process(text)
            if remove_stop_words:
                text = ' '.join([x for x in text.split() if x not in stop_words])
            train_text.append(text)
    emo = data.get_target()  # returns target

    return train_text, emo, emo_list, NUM_CLASS


def emoset_data(file_path='data/EmoSetProcessedEkmanNoDupSingle.csv', remove_stop_words=True, get_text=True):
    NUM_CLASS = 6
    emo_list = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust']
    emo_dict = dict(zip(emo_list, range(len(emo_list))))
    stop_words = set(stopwords.words('english'))

    label = []
    train_text = []
    df = pd.read_csv(file_path)
    for i, row in df.iterrows():
        if get_text:
            text = str(row['tweet'])
            if remove_stop_words:
                text = ' '.join([x for x in str(text).split() if x not in stop_words])
            train_text.append(text)

        emo = row['emo'].strip()
        emo_idx = emo_dict[emo]
        label.append(emo_idx)

    return train_text, label, emo_list, NUM_CLASS


def tec_data(file_path='data/TEC.txt', remove_stop_words=True, get_text=True):
    NUM_CLASS = 6
    emo_list = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust']

    stop_words = set(stopwords.words('english'))

    f = open(file_path, 'r', encoding='utf8')
    lines = f.readlines()
    f.close()

    label = []
    train_text = []
    for line in lines:
        text, emo = line.split('\t')
        if get_text:
            from utils.tweet_processor import tweet_process
            text = tweet_process(text)
            if remove_stop_words:
                text = ' '.join([x for x in text.split() if x not in stop_words])
            train_text.append(text)
        label.append(int(emo))
    return train_text, label, emo_list, NUM_CLASS
