
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
PARLAI_PAD_TOKEN = "__PAD__"
EMPTYPERSONA_TOKEN = "[PER]"
START_OF_COMMENT = "[SOC]"
END_OF_COMMENT = "[EOC]"

UNUSED_BERT_TOKEN_1 = "[unused1]"
UNUSED_BERT_TOKEN_2 = "[unused2]"
UNUSED_BERT_TOKEN_3 = "[unused3]"

def tokenize(text, tokenizer):
    return tokenizer.tokenize(text.strip())



