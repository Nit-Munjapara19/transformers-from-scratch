import torch
from torch.utils.data import Dataset
from tokenization import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, text_to_token_ids

class TranslationDataset(Dataset):
    def __init__(self, sentence_pairs, english_vocabulary, hindi_vocabulary, max_sequence_length):
        self.sentence_pairs = sentence_pairs
        self.english_vocabulary = english_vocabulary
        self.hindi_vocabulary = hindi_vocabulary
        self.max_sequence_length = max_sequence_length

    def encode_english(self, sentence):
        token_ids = text_to_token_ids(sentence, "english", self.english_vocabulary)
        token_ids = token_ids[:self.max_sequence_length - 1] + [self.english_vocabulary[EOS_TOKEN]]
        padding_needed = self.max_sequence_length - len(token_ids)
        token_ids += [self.english_vocabulary[PAD_TOKEN]] * padding_needed
        return token_ids

    def encode_hindi(self, sentence):
        token_ids = text_to_token_ids(sentence, "hindi", self.hindi_vocabulary)
        token_ids = [self.hindi_vocabulary[BOS_TOKEN]] + token_ids[:self.max_sequence_length - 2] + [self.hindi_vocabulary[EOS_TOKEN]]
        padding_needed = self.max_sequence_length - len(token_ids)
        token_ids += [self.hindi_vocabulary[PAD_TOKEN]] * padding_needed
        return token_ids

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, index):
        english_sentence, hindi_sentence = self.sentence_pairs[index]
        english_ids = self.encode_english(english_sentence)
        hindi_ids = self.encode_hindi(hindi_sentence)
        decoder_input_ids = hindi_ids[:-1]
        target_ids = hindi_ids[1:]

        return {
            "encoder_input": torch.tensor(english_ids, dtype=torch.long),
            "decoder_input": torch.tensor(decoder_input_ids, dtype=torch.long),
            "target": torch.tensor(target_ids, dtype=torch.long)
        }