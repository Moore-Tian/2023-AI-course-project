def word2features(sentence, idx):
    word = sentence[idx]
    # 当前词语的基础特征
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
        }

    # 添加上一个词的特征
    if idx > 0:
        word1 = sentence[idx - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:]
        })
    else:
        features['FIRST'] = True

    # 添加上上个词的特征
    if idx > 1:
        word2 = sentence[idx - 2]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.isdigit()': word2.isdigit(),
            '-2:word[-3:]': word2[-3:],
            '-2:word[-2:]': word2[-2:]
        })

    # 添加上上上个词的特征
    if idx > 2:
        word3 = sentence[idx - 3]
        features.update({
            '-3:word.lower()': word3.lower(),
            '-3:word.istitle()': word3.istitle(),
            '-3:word.isupper()': word3.isupper(),
            '-3:word.isdigit()': word3.isdigit(),
            '-3:word[-3:]': word3[-3:],
            '-3:word[-2:]': word3[-2:]
        })

    # 添加后一个词的特征
    if idx < len(sentence) - 1:
        word1 = sentence[idx + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word[-3:]': word1[-3:],
            '+1:word[-2:]': word1[-2:]
        })
    else:
        features['_FIEST'] = True

    # 添加后后一个词的特征
    if idx < len(sentence) - 2:
        word2 = sentence[idx + 2]
        features.update({
            '+2:word.lower()': word2.lower(),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.isdigit()': word2.isdigit(),
            '+2:word[-3:]': word2[-3:],
            '+2:word[-2:]': word2[-2:]
        })

    # 添加后后后一个词的特征
    if idx < len(sentence) - 3:
        word3 = sentence[idx + 3]
        features.update({
            '+3:word.lower()': word3.lower(),
            '+3:word.istitle()': word3.istitle(),
            '+3:word.isupper()': word3.isupper(),
            '+3:word.isdigit()': word3.isdigit(),
            '+3:word[-3:]': word3[-3:],
            '+3:word[-2:]': word3[-2:]
        })

    if idx == 1:
        features['SECOND'] = True
    elif idx == 2:
        features['SECOND'] = True
    else:
        features['MID'] = True

    if idx == len(sentence) - 2:
        features['_SECOND'] = True
    elif idx == len(sentence) - 3:
        features['_THIRD'] = True
    else:
        features['_MID'] = True

    return features


def get_features(dataset):
    features = []
    labels = []
    for words, tags in dataset:
        features.append([word2features(words, i) for i in range(len(words))])
        labels.append(tags)
    return features, labels
