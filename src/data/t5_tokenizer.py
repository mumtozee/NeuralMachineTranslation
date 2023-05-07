class T5Tokenizer:
    def __init__(self, toker):
        self.toker = toker
        self.special_token_set = {0, 1}
    
    def decode(self, token_list):
        res = []
        return self.toker.tokenize(self.toker.decode(token_list, skip_special_tokens=True))
        # for t in token_list:
        #     if t not in self.special_token_set:
        #         res.append(self.toker.convert_ids_to_tokens([t])[0])
        return res
