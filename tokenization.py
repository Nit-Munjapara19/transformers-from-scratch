import re
from collections import Counter

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

def normalize_text(text):
    text = text.strip()
    text = re.sub(r"([.,!?;:।])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_english(text):
    return normalize_text(text).lower().split()

def tokenize_hindi(text):
    return normalize_text(text).split()

def build_vocabulary(sentence_pairs, language, minimum_frequency=1):
    token_counter = Counter()

    for english_sentence, hindi_sentence in sentence_pairs:
        if language == "english":
            tokens = tokenize_english(english_sentence)
        elif language == "hindi":
            tokens = tokenize_hindi(hindi_sentence)
        else:
            raise ValueError("language must be 'english' or 'hindi'")

        token_counter.update(tokens)

    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}

    for token, frequency in token_counter.most_common():
        if frequency >= minimum_frequency and token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    return vocabulary

def build_reverse_vocabulary(vocabulary):
    return {index: token for token, index in vocabulary.items()}

def text_to_token_ids(text, language, vocabulary):
    if language == "english":
        tokens = tokenize_english(text)
    elif language == "hindi":
        tokens = tokenize_hindi(text)
    else:
        raise ValueError("language must be 'english' or 'hindi'")

    unknown_token_id = vocabulary[UNK_TOKEN]
    return [vocabulary.get(token, unknown_token_id) for token in tokens]

def token_ids_to_text(token_ids, id_to_token):
    tokens = []

    for token_id in token_ids:
        token = id_to_token.get(int(token_id), UNK_TOKEN)

        if token in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}:
            continue

        tokens.append(token)

    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:।])", r"\1", text)

    return text.strip()