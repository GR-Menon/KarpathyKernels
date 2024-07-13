from .tok_base import get_stats, merge_pair, Tokenizer


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256

        num_merges = vocab_size - 256

        byte_stream = text.encode('utf-8')
        ids = list(byte_stream)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            merge_stats = get_stats(ids)

            pair = max(merge_stats, key=merge_stats.get)

            idx = 256 + i
            ids = merge_pair(ids, pair, idx)

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(
                    f"merge [{i + 1}/{num_merges}]: {pair} -> {idx} ({vocab[idx]}) had ({merge_stats[pair]}) occurrences")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        byte_stream = text.encode('utf-8')
        tokens = list(byte_stream)

        while len(tokens) >= 2:
            token_stats = get_stats(tokens)
            pair = min(token_stats, key=lambda p: self.merges.get(p, float('inf')))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            tokens = merge_pair(tokens, pair, idx)
        return tokens

    def decode(self, tokens):
        tokens = b"".join(self.vocab[token] for token in tokens)
        text = tokens.decode('utf-8', errors='replace')
        return text
