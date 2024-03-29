from typing import Any


en_tag_mapping = {
    # not a entity
    "O": 0,
    # person entities
    "B-PER": 1,
    "I-PER": 2,
    # organization entities
    "B-ORG": 3,
    "I-ORG": 4,
    # location entities
    "B-LOC": 5,
    "I-LOC": 6,
    # other entities
    "B-MISC": 7,
    "I-MISC": 8,
    "<PAD>" : 9
}
num_en_tag = len(en_tag_mapping)

cn_tag_mapping = {
    'O': 0,
    'B-NAME': 1, 'M-NAME': 2, 'E-NAME': 3, 'S-NAME': 4,
    'B-CONT': 5, 'M-CONT': 6, 'E-CONT': 7, 'S-CONT': 8,
    'B-EDU': 9,  'M-EDU': 10, 'E-EDU': 11, 'S-EDU': 12,
    'B-TITLE': 13, 'M-TITLE': 14, 'E-TITLE': 15, 'S-TITLE': 16,
    'B-ORG': 17, 'M-ORG': 18, 'E-ORG': 19, 'S-ORG': 20,
    'B-RACE': 21, 'M-RACE': 22, 'E-RACE': 23, 'S-RACE': 24,
    'B-PRO': 25, 'M-PRO': 26, 'E-PRO': 27, 'S-PRO': 28,
    'B-LOC': 29, 'M-LOC': 30, 'E-LOC': 31, 'S-LOC': 32,
    '<PAD>': 33
}
num_cn_tag = len(cn_tag_mapping)


class tag_mapping():
    def __init__(self, language):
        if language == 'English':
            self.encode_mapping = en_tag_mapping
            self.num_tag = num_en_tag
        elif language == 'Chinese':
            self.encode_mapping = cn_tag_mapping
            self.num_tag = num_cn_tag
        
        self.decode_mapping = {value: key for key, value in self.encode_mapping.items()}

    def encode(self, tags):
        return [self.encode_mapping[tag] for tag in tags]

    def decode(self, codes):
        return [self.decode_mapping[code] for code in codes]

    def __len__(self):
        return self.num_tag