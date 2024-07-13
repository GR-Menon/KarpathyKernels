import os
import time

from tokenizers import BasicTokenizer

text = open('train_corpus.txt', 'r', encoding='utf-8').read()

os.makedirs("models", exist_ok=True)

start_t = time.time()

for Tokenizer, name in zip([BasicTokenizer], ["basic"]):
    tokenizer = Tokenizer()
    tokenizer.train(text=text, vocab_size=512, verbose=True)

    prefix = os.path.join("models", name)
    tokenizer.save(prefix)

end_t = time.time()

print(f"Training time: {end_t - start_t:.2f} seconds")
