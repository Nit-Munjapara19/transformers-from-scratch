from torch.utils.data import DataLoader
from data import load_translation_data
from dataset import TranslationDataset
from tokenization import (build_vocabulary,build_reverse_vocabulary)

BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 30

training_pairs, validation_pairs = load_translation_data()

print("Training pairs:", len(training_pairs))
print("Validation pairs:", len(validation_pairs))

english_vocabulary = build_vocabulary(
    training_pairs,
    language="english",
    minimum_frequency=1
)
hindi_vocabulary = build_vocabulary(
    training_pairs,
    language="hindi",
    minimum_frequency=1
)

english_id_to_token = build_reverse_vocabulary(
    english_vocabulary
)
hindi_id_to_token = build_reverse_vocabulary(
    hindi_vocabulary
)


training_dataset = TranslationDataset(
    sentence_pairs=training_pairs,
    english_vocabulary=english_vocabulary,
    hindi_vocabulary=hindi_vocabulary,
    max_sequence_length=MAX_SEQUENCE_LENGTH
)
validation_dataset = TranslationDataset(
    sentence_pairs=validation_pairs,
    english_vocabulary=english_vocabulary,
    hindi_vocabulary=hindi_vocabulary,
    max_sequence_length=MAX_SEQUENCE_LENGTH
)

training_loader = DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

batch = next(iter(training_loader))

print("\nEncoder input shape:")
print(batch["encoder_input"].shape)

print("\nDecoder input shape:")
print(batch["decoder_input"].shape)

print("\nTarget shape:")
print(batch["target"].shape)