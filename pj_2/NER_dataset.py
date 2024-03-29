class NER_dataset():
    def __init__(self, datapath) -> None:
        super().__init__()
        self.sentences = []
        self.word_mapping = None
        self.tag_mapping = None

        with open(datapath, 'r') as f:
            sentence_words = []
            sentence_tags = []
            for line in f:
                line = line.strip()
                if not line:
                    self.sentences.append((sentence_words, sentence_tags))
                    sentence_words = []
                    sentence_tags = []
                else:
                    word, tag = line.split()
                    sentence_words.append(word)
                    sentence_tags.append(tag)

        if sentence_words:
            self.sentences.append((sentence_words, sentence_tags))

    def get_tag_mapping(self, tag_mapping):
        self.tag_mapping = tag_mapping

    def get_word_mapping(self, word_record):
        self.word_mapping = word_record

    def __getitem__(self, idx):
        sentence_words, sentence_tags = self.sentences[idx]
        if self.word_mapping is not None:
            sentence_words = [self.word_mapping[word] if word in self.word_mapping else 0 for word in sentence_words]
        if self.tag_mapping is not None:
            sentence_tags = [self.tag_mapping[tag] for tag in sentence_tags]
        return sentence_words, sentence_tags

    def __len__(self):
        return len(self.sentences)
