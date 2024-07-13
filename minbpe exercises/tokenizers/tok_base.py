import unicodedata


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(ids, pair, idx):
    new_ids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def remove_control_chars(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != 'C':
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)


def print_token(token: bytes) -> str:
    s = token.decode("utf-8", errors="replace")
    return remove_control_chars(s)


class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = self.build_vocab()

    def train(self, text, vocab_size, verbose=False) -> None:
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, tokens):
        raise NotImplementedError

    def build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def save(self, prefix):
        model_file = prefix + ".model"

        with open(model_file, 'w') as f:
            f.write('minbpe v1\n')

            for idx1, idx2 in self.merges.items():
                f.write(f"{idx1} {idx2}\n")

        vocab_file = prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w') as f:
            for idx, token in self.vocab.items():
                s = print_token(token)

                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = print_token(self.vocab[idx0])
                    s1 = print_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> {idx}\n")
                else:
                    f.write(f"[{s}] -> {idx}\n")
