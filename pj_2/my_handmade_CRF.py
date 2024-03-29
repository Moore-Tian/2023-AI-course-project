import numpy as np
from collections import defaultdict
from tqdm import tqdm

class feature_functions():
    def __init__(self, data_set):
        self.Unigram_currentword = defaultdict(int)
        self.Uni_curr_size = 0
        self.Unigram_previousword = defaultdict(int)
        self.Uni_pre_size = 0
        self.Unigram_nextword = defaultdict(int)
        self.Uni_next_size = 0
        self.Unigram_neighborword = defaultdict(int)
        self.Uni_neig_size = 0
        self.Bigram_currentword = defaultdict(int)
        self.Bi_curr_size = 0
        self.Bigram_previousword = defaultdict(int)
        self.Bi_pre_size = 0
        self.Bigram_nextword = defaultdict(int)
        self.Bi_next_size = 0
        self.Bigram_neighborword = defaultdict(int)
        self.Bi_neig_size = 0

        self.padding = 1
        self.record = defaultdict(int)
        self.threshold = 1
        self.features_init(data_set)

    # 通过统计训练集来获得特征函数
    def features_init(self, data_set):
        for words, tags in tqdm(data_set):
            words = words[:]
            tags = tags[:]
            words.insert(0, -1)
            words.append(-1)
            tags.append(-1)
            for i in range(self.padding, len(words) - 1):
                if 1:
                    if ((words[i]), (tags[i - 1])) not in self.Unigram_currentword:
                        if self.record[((words[i]), (tags[i - 1]))] >= self.threshold:
                            self.Unigram_currentword[((words[i]), (tags[i - 1]))] = 0
                            self.Uni_curr_size += 1
                        else:
                            self.record[((words[i]), (tags[i - 1]))] += 1
                    if ((words[i], words[i - 1]), (tags[i - 1])) not in self.Unigram_previousword:
                        if self.record[((words[i], words[i - 1]), (tags[i - 1]))] >= self.threshold:
                            self.Unigram_previousword[((words[i], words[i - 1]), (tags[i - 1]))] = 0
                            self.Uni_pre_size += 1
                        else:
                            self.record[((words[i], words[i - 1]), (tags[i - 1]))] += 1
                    if ((words[i], words[i + 1]), (tags[i - 1])) not in self.Unigram_nextword:
                        if self.record[((words[i], words[i + 1]), (tags[i - 1]))] >= self.threshold:
                            self.Unigram_nextword[((words[i], words[i + 1]), (tags[i - 1]))] = 0
                            self.Uni_next_size += 1
                        else:
                            self.record[((words[i], words[i + 1]), (tags[i - 1]))] += 1
                if 1:
                    if ((words[i]), (tags[i - 1], tags[i])) not in self.Bigram_currentword:
                        if self.record[((words[i]), (tags[i - 1], tags[i]))] >= self.threshold:
                            self.Bigram_currentword[((words[i]), (tags[i - 1], tags[i]))] = 0
                            self.Bi_curr_size += 1
                        else:
                            self.record[((words[i]), (tags[i - 1], tags[i]))] += 1
                    if ((words[i], words[i - 1]), (tags[i - 1], tags[i])) not in self.Bigram_previousword:
                        if self.record[((words[i], words[i - 1]), (tags[i - 1], tags[i]))] >= self.threshold:
                            self.Bigram_previousword[((words[i], words[i - 1]), (tags[i - 1], tags[i]))] = 0
                            self.Bi_pre_size += 1
                        else:
                            self.record[((words[i], words[i - 1]), (tags[i - 1], tags[i]))] += 1
                    if ((words[i], words[i + 1]), (tags[i - 1], tags[i])) not in self.Bigram_nextword:
                        if self.record[((words[i], words[i + 1]), (tags[i - 1], tags[i]))] >= self.threshold:
                            self.Bigram_nextword[((words[i], words[i + 1]), (tags[i - 1], tags[i]))] = 0
                            self.Bi_next_size += 1
                        else:
                            self.record[((words[i], words[i + 1]), (tags[i - 1], tags[i]))] += 1


    def compute_features(self, words, tag_1, tag_2, i):
        total_val = self.Unigram_currentword[((words[i]), (tag_1))] + \
            self.Unigram_previousword[((words[i], words[i - 1]), (tag_1))] + \
                self.Unigram_nextword[((words[i], words[i + 1]), (tag_1))] + \
                        self.Bigram_currentword[((words[i]), (tag_1, tag_2))] + \
                            self.Bigram_previousword[((words[i], words[i - 1]), (tag_1, tag_2))] + \
                                self.Bigram_nextword[((words[i], words[i + 1]), (tag_1, tag_2))]

        return total_val

    def update(self, words, pred_tags, tags, lr):

        words = words[:]
        tags = tags[:]
        pred_tags = pred_tags[:]
        words.insert(0, -1)
        words.append(-1)
        tags.append(-1)
        pred_tags.append(-1)
        record = 0

        for i in range(self.padding, len(words) - 1):
            if tags[i - 1] == pred_tags[i - 1]:
                pass
            else:
                if ((words[i]), (tags[i - 1])) in self.Unigram_currentword:
                    self.Unigram_currentword[((words[i]), (tags[i - 1]))] += lr
                    record += 1
                if ((words[i]), (pred_tags[i - 1])) in self.Unigram_currentword:
                    self.Unigram_currentword[((words[i]), (pred_tags[i - 1]))] -= lr
                    record += 1

                if ((words[i], words[i - 1]), (tags[i - 1])) in self.Unigram_previousword:
                    self.Unigram_previousword[((words[i], words[i - 1]), (tags[i - 1]))] += lr
                    record += 1
                if (words[i], words[i - 1], pred_tags[i - 1]) in self.Unigram_previousword:
                    self.Unigram_previousword[((words[i], words[i - 1]), (pred_tags[i - 1]))] -= lr
                    record += 1

                if ((words[i], words[i + 1]), (tags[i - 1])) in self.Unigram_nextword:
                    self.Unigram_nextword[((words[i], words[i + 1]), (tags[i - 1]))] += lr
                    record += 1
                if (words[i], words[i + 1], pred_tags[i - 1]) in self.Unigram_nextword:
                    self.Unigram_nextword[((words[i], words[i + 1]), (pred_tags[i - 1]))] -= lr
                    record += 1

            if tags[i] == pred_tags[i] and tags[i - 1] == pred_tags[i - 1]:
                pass
            else:
                if ((words[i]), (tags[i - 1], tags[i])) in self.Bigram_currentword:
                    self.Bigram_currentword[((words[i]), (tags[i - 1], tags[i]))] += lr
                    record += 1
                if ((words[i]), (pred_tags[i - 1], pred_tags[i])) in self.Bigram_currentword:
                    self.Bigram_currentword[((words[i]), (pred_tags[i - 1], pred_tags[i]))] -= lr
                    record += 1

                if ((words[i], words[i - 1]), (tags[i - 1], tags[i])) in self.Bigram_previousword:
                    self.Bigram_previousword[((words[i], words[i - 1]), (tags[i - 1], tags[i]))] += lr
                    record += 1
                if ((words[i], words[i - 1]), (pred_tags[i - 1], pred_tags[i])) in self.Bigram_previousword:
                    self.Bigram_previousword[((words[i], words[i - 1]), (pred_tags[i - 1], pred_tags[i]))] -= lr
                    record += 1

                if ((words[i], words[i + 1]), (tags[i - 1], tags[i])) in self.Bigram_nextword:
                    self.Bigram_nextword[((words[i], words[i + 1]), (tags[i - 1], tags[i]))] += lr
                    record += 1
                if ((words[i], words[i + 1]), (pred_tags[i - 1], pred_tags[i])) in self.Bigram_nextword:
                    self.Bigram_nextword[((words[i], words[i + 1]), (pred_tags[i - 1], pred_tags[i]))] -= lr
                    record += 1


class CRF():
    def __init__(self, feature_functions, num_states):
        self.features = feature_functions
        self.num_states = num_states
        self.padding = 1
    
    def decode(self, words):
        length = len(words)
        words = words[:]
        words.insert(0, -1)
        words.append(-1)
        path_scores = np.zeros((length + 1, self.num_states))
        path_record = np.zeros((length, self.num_states))
        for word in range(length):
            for curr_state in range(self.num_states):
                for next_state in range(self.num_states):
                    score = path_scores[word, curr_state]
                    score += self.features.compute_features(words, curr_state, next_state, word)
                    if score > path_scores[word + 1, next_state]:
                        path_scores[word + 1, next_state] = score
                        path_record[word, next_state] = curr_state
        tags = np.zeros(length)
        tags[length - 1] = np.argmax(path_scores[length])
        for i in range(length - 1):
            tags[int(length - 2 - i)] = path_record[int(length - 1 - i), int(tags[int(length - 1 - i)])]
            
        return tags

    def train(self, train_data, max_epoch=2, lr=1):
        train_acc = []
        for epoch in range(max_epoch):
            total_pred = 0
            correct_pred = 0
            for words, tags in tqdm(train_data):
                total_pred += len(words)
                pred_tags = self.decode(words)
                correct_pred += np.sum(pred_tags == tags)
                self.features.update(words, pred_tags.tolist(), tags, lr)

            train_acc.append(correct_pred / total_pred)
            print("epoch: {}, train_acc: {}".format(epoch + 1, train_acc[-1]))