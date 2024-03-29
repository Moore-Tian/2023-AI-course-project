import numpy as np
from NER_dataset import *
from tag_mapping import *
from word_mapping import *
from tqdm import tqdm
from utils import *
from check import *


class HMM():
    def __init__(self, num_state, num_words):
        self.num_state = num_state
        self.num_words = num_words
        # 该初识化在本任务中并无作用，但在用EM算法进行参数估计时会有用
        self.initial_probability = np.random.rand(num_state)
        self.initial_probability /= np.sum(self.initial_probability)

        self.transition_probability = np.random.rand(num_state, num_state)
        self.transition_probability /= np.sum(self.transition_probability, axis=1, keepdims=True)

        self.emission_probability = np.random.rand(num_state, num_words)
        self.emission_probability /= np.sum(self.emission_probability, axis=1, keepdims=True)

    # 通过对训练集的简单统计得到初始概率、转移概率和发射概率
    def param_estimate(self, dataset, smoothing_factor=1):
        self.initial_probability = smoothing_factor * np.ones_like(self.initial_probability)
        self.transition_probability = smoothing_factor * np.ones_like(self.transition_probability)
        self.emission_probability = smoothing_factor * np.ones_like(self.emission_probability)

        for words, tags in dataset:
            self.initial_probability[tags[0]] += 1
            for i in range(len(tags) - 1):
                self.transition_probability[tags[i], tags[i + 1]] += 1
                self.emission_probability[tags[i], words[i]] += 1
            self.emission_probability[tags[-1], words[-1]] += 1

        self.initial_probability /= np.sum(self.initial_probability)
        self.transition_probability /= np.sum(self.transition_probability, axis=1, keepdims=True)
        self.emission_probability /= np.sum(self.emission_probability, axis=1, keepdims=True)

    def evaluate(self, words):
        length = len(words)
        # prob[t,i] 为前 t 项观测值符合观测序列且第t项的状态为i的情况所对应的概率
        prob = np.zeros((length + 1, self.num_state))
        prob[0] = self.initial_probability
        for t in range(length):
            for current_state in range(self.num_state):
                for previous_state in range(self.num_state):
                    prob[t + 1, current_state] += prob[t, previous_state] * self.transition_probability[previous_state, current_state] * \
                                                self.emission_probability[previous_state, words[t]]

        return np.sum(prob[-1])

    def decode(self, words):
        length = len(words)

        # 使用概率原值进行累乘，会使结果过小，导致下溢，故此处将其转换到对数空间上进行运算
        delta = np.full((length, self.num_state), -np.inf)
        psi = np.zeros((length, self.num_state), dtype=int)

        delta[0] = np.log(self.initial_probability) + np.log(self.emission_probability[:, words[0]])
        for t in range(1, length):
            for current_state in range(self.num_state):
                prob = delta[t - 1] + np.log(self.transition_probability[:, current_state]) + np.log(self.emission_probability[current_state, words[t]])
                delta[t, current_state] = np.max(prob)
                psi[t, current_state] = np.argmax(prob)

        states = np.zeros(length, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(length - 1, 0, -1):
            states[t - 1] = psi[t, states[t]]

        return states

    def save(self, file_name):
        np.savez(file_name, transition_probability=self.transition_probability, emission_probability=self.emission_probability,
                 initial_probability=self.initial_probability)

    def load(self, file_name):
        npzfile = np.load(file_name)
        self.transition_probability = npzfile['transition_probability']
        self.emission_probability = npzfile['emission_probability']
        self.initial_probability = npzfile['initial_probability']


# 该模型的测试函数
def HMM_check(language):
    train_data, valid_data = get_data_set(language)

    TagMapping = tag_mapping(language)
    WordMapping = word_mapping(train_data)

    # 为数据集装载映射
    train_data.get_tag_mapping(TagMapping.encode_mapping)
    train_data.get_word_mapping(WordMapping.encode_mapping)
    valid_data.get_tag_mapping(TagMapping.encode_mapping)
    valid_data.get_word_mapping(WordMapping.encode_mapping)

    # 建立模型并初始化
    model = HMM(TagMapping.num_tag, WordMapping.num_word)
    smoothing_factor = 1
    model.param_estimate(train_data, smoothing_factor=smoothing_factor)

    # 输出
    my_path = f"my_{language}_HMM_result.txt"
    file = open(my_path, "w")
    for words, tags in tqdm(valid_data):
        tags_pred = model.decode(words)
        words_decoded = WordMapping.decode(words)
        tags_decoded = TagMapping.decode(tags_pred)
        for i in range(len(words)):
            file.write(f'{words_decoded[i]} {tags_decoded[i]}\n')
        file.write('\n')
    file.close()

    gold_path = f"./NER/{language}/validation.txt"
    check(language, gold_path, my_path)
