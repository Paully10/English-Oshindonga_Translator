import json
import re
import sys
from collections import Counter
from pathlib import Path


# Make the AURA project root importable when this file is
# executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]


def basic_tokenize(text):
    """
    Convert text into simple word/punctuation tokens.
    """
    return re.findall(
        r"\w+|[^\w\s]",
        text.lower(),
        flags=re.UNICODE
    )


class Vocabulary:

    def __init__(self, min_freq=1, max_size=None):
        self.min_freq = min_freq
        self.max_size = max_size

        self.itos = []
        self.stoi = {}

    def build(self, texts):
        """
        Build vocabulary from a collection of texts.
        """

        counter = Counter()

        for text in texts:
            counter.update(basic_tokenize(text))

        words = [
            word
            for word, frequency in counter.most_common()
            if frequency >= self.min_freq
        ]

        if self.max_size:
            available_slots = max(
                0,
                self.max_size - len(SPECIAL_TOKENS)
            )

            words = words[:available_slots]

        self.itos = SPECIAL_TOKENS + words

        self.stoi = {
            token: index
            for index, token in enumerate(self.itos)
        }

    def encode(self, text, add_bos=True, add_eos=True):

        tokens = basic_tokenize(text)

        ids = []

        if add_bos:
            ids.append(self.stoi[BOS])

        for token in tokens:
            ids.append(
                self.stoi.get(
                    token,
                    self.stoi[UNK]
                )
            )

        if add_eos:
            ids.append(self.stoi[EOS])

        return ids

    def decode(self, ids):

        tokens = []

        for index in ids:

            if index < 0 or index >= len(self.itos):
                continue

            token = self.itos[index]

            if token in SPECIAL_TOKENS:
                continue

            tokens.append(token)

        text = " ".join(tokens)

        # Fix spacing before punctuation.
        text = re.sub(
            r"\s+([,.!?;:])",
            r"\1",
            text
        )

        return text

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def bos_id(self):
        return self.stoi[BOS]

    @property
    def eos_id(self):
        return self.stoi[EOS]

    def __len__(self):
        return len(self.itos)

    def save(self, path):

        data = {
            "itos": self.itos,
            "min_freq": self.min_freq,
            "max_size": self.max_size
        }

        Path(path).write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, path):

        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        vocabulary = cls(
            data["min_freq"],
            data["max_size"]
        )

        vocabulary.itos = data["itos"]

        vocabulary.stoi = {
            token: index
            for index, token in enumerate(
                vocabulary.itos
            )
        }

        return vocabulary


def main():

    from config import (
        TRAIN_FILE,
        MIN_FREQ,
        MAX_VOCAB_SIZE,
        TOKENIZER_FILE
    )

    print("=" * 50)
    print("AURA TOKENIZER")
    print("=" * 50)

    print(f"Reading training data:")
    print(TRAIN_FILE)

    texts = []

    with open(
        TRAIN_FILE,
        encoding="utf-8"
    ) as file:

        for line in file:

            row = json.loads(line)

            texts.append(row["source"])
            texts.append(row["target"])

    print(f"Texts processed: {len(texts):,}")

    vocabulary = Vocabulary(
        min_freq=MIN_FREQ,
        max_size=MAX_VOCAB_SIZE
    )

    vocabulary.build(texts)

    vocabulary.save(TOKENIZER_FILE)

    print()
    print(f"Vocabulary size: {len(vocabulary):,}")
    print(f"PAD token: {vocabulary.pad_id}")
    print(f"UNK token: {vocabulary.stoi[UNK]}")
    print(f"BOS token: {vocabulary.bos_id}")
    print(f"EOS token: {vocabulary.eos_id}")

    print()
    print("Vocabulary saved to:")
    print(TOKENIZER_FILE)

    print()
    print("Tokenizer test:")

    test_sentence = "I am going to school."

    encoded = vocabulary.encode(test_sentence)
    decoded = vocabulary.decode(encoded)

    print("Input:   ", test_sentence)
    print("Encoded: ", encoded)
    print("Decoded: ", decoded)

    print()
    print("Tokenizer completed successfully.")
    print("=" * 50)


if __name__ == "__main__":
    main()