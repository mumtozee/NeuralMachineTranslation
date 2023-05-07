class SpaceTokenizer:
    def __init__(self, sentence_list, pad_flag):
        self.pad_flag = pad_flag
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK", 3: "PAD", 4: " "}
        self.word2index = {"SOS": 0, "EOS": 1, "UNK": 2, 'PAD': 3, " ": 4}
        self.n_words = len(self.word2index)
        self.max_sent_len = -1
        self.special_tokens_set = {'SOS', 'EOS', 'PAD'}

        for sent in sentence_list:
            token_list = sent.split()
            sent_words_amount = len(token_list)
            if sent_words_amount > self.max_sent_len:
                self.max_sent_len = sent_words_amount

            for token in token_list:
                if token not in self.word2index:
                    self.word2index[token] = self.n_words
                    self.word2count[token] = 1
                    self.index2word[self.n_words] = token
                    self.n_words += 1
                else:
                    self.word2count[token] += 1

        self.max_sent_len += 2# add EOS/SOS tokens

        print(f'Space tokenizer fitted - {len(self.word2index)} tokens')

    def pad_sent(self, token_ids_list):
        if len(token_ids_list) < self.max_sent_len:
            padded_token_ids_list = token_ids_list + [self.word2index['PAD']] * (self.max_sent_len - len(token_ids_list))
        else:
            padded_token_ids_list = token_ids_list[:self.max_sent_len - 1] + [self.word2index['EOS']]
        return padded_token_ids_list

    def __call__(self, sentence):
        tokenized_data = self.tokenize(sentence)
        if self.pad_flag:
            tokenized_data = self.pad_sent(tokenized_data)
        return tokenized_data

    def tokenize(self, sentence):
        tokenized_data = []
        tokenized_data.append(self.word2index['SOS'])
        for word in sentence.split():
            if word in self.word2index:
                tokenized_data.append(self.word2index[word])
            else:
                tokenized_data.append(self.word2index['UNK'])
        tokenized_data.append(self.word2index['EOS'])
        return tokenized_data

    def decode(self, token_list):
        predicted_tokens = []

        for token_id in token_list:
            predicted_token = self.index2word[token_id]
            predicted_tokens.append(predicted_token)
        filtered_tokens = list(filter(lambda x: x not in self.special_tokens_set, predicted_tokens))

        return filtered_tokens