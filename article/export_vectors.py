"""Export GloVe vectors as binary files for browser consumption.

Produces three sizes:
  - glove-small.bin   (10K words, ~4MB)
  - glove-medium.bin  (50K words, ~20MB)
  - glove-large.bin   (400K words, ~160MB)

Binary format:
  - 4 bytes: uint32 num_words
  - 4 bytes: uint32 num_dims
  - For each word:
    - 2 bytes: uint16 word_length
    - word_length bytes: UTF-8 encoded word
  - num_words * num_dims * 4 bytes: float32 vectors (row-major)

This layout lets the browser read the vocabulary first (fast) and then
stream/read the vectors as a single typed array.
"""

import struct
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'


def load_glove(path='data/glove.6B.100d.txt'):
    words = []
    vecs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            words.append(parts[0])
            vecs.append([float(x) for x in parts[1:]])
    return words, np.array(vecs, dtype=np.float32)


def export(words, vecs, outpath, n=None, must_have=None):
    word_to_idx = {w: i for i, w in enumerate(words)}
    if n is not None:
        selected = list(range(min(n, len(words))))
        if must_have:
            in_set = set(selected)
            for w in must_have:
                idx = word_to_idx.get(w)
                if idx is not None and idx not in in_set:
                    selected.append(idx)
                    in_set.add(idx)
        words = [words[i] for i in selected]
        vecs = vecs[selected]
    num_words = len(words)
    num_dims = vecs.shape[1]

    with open(outpath, 'wb') as f:
        # Header
        f.write(struct.pack('<II', num_words, num_dims))
        # Vocabulary
        for w in words:
            encoded = w.encode('utf-8')
            f.write(struct.pack('<H', len(encoded)))
            f.write(encoded)
        # Vectors as contiguous float32
        f.write(vecs.tobytes())

    size_mb = outpath.stat().st_size / (1024 * 1024)
    print(f'{outpath.name}: {num_words} words, {num_dims}d, {size_mb:.1f}MB')


def main():
    # Try loading from npz first (faster), fall back to text
    npz = Path('vecs.npz')
    txt = Path('data/glove.6B.100d.txt')

    if npz.exists():
        print('Loading from vecs.npz...')
        with np.load(npz) as data:
            vecs = data['vec'].astype(np.float32)
            words = list(data['voc'])
    elif txt.exists():
        print('Loading from glove text file...')
        words, vecs = load_glove(str(txt))
    else:
        print('No vector file found. Need vecs.npz or data/glove.6B.100d.txt')
        return

    # Words used by the article that may fall outside smaller vocabulary cutoffs
    article_words = {
        # superlatives
        'poor', 'poorer', 'poorest', 'rich', 'richer', 'richest',
        'short', 'shorter', 'shortest', 'slow', 'slower', 'slowest',
        'fast', 'faster', 'fastest', 'soft', 'softer', 'softest',
        'strong', 'stronger', 'strongest', 'mean', 'meaner', 'meanest',
        'dark', 'darker', 'darkest', 'smart', 'smarter', 'smartest',
        # numbers
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
        'nine', 'ten', 'eleven', 'twelve', 'thirteen',
        # gendered pairs
        'she', 'he', 'woman', 'man', 'herself', 'himself', 'her', 'him',
        'hers', 'his', 'gal', 'guy', 'girl', 'boy', 'girls', 'boys',
        'female', 'male', 'females', 'males',
        'king', 'queen', 'actor', 'actress', 'dad', 'mom', 'father', 'mother',
        'brother', 'sister', 'uncle', 'aunt', 'heir', 'heiress', 'duke',
        'duchess', 'nephew', 'niece', 'sir', 'madame', 'masculine', 'feminine',
        # professions
        'caretaker', 'homemaker', 'doctor', 'nurse', 'programmer', 'teacher',
        'wife', 'husband', 'soldier', 'salesperson', 'analyst', 'therapist',
        'trainer', 'instructor', 'ceo', 'assistant', 'telemarketer',
        'bartender', 'clerk', 'designer', 'scientist', 'manager', 'boss',
        'employee',
    }

    DATA_DIR.mkdir(exist_ok=True)

    export(words, vecs, DATA_DIR / 'glove-small.bin', n=10_000, must_have=article_words)
    export(words, vecs, DATA_DIR / 'glove-medium.bin', n=50_000, must_have=article_words)
    export(words, vecs, DATA_DIR / 'glove-large.bin', n=len(words))


if __name__ == '__main__':
    main()
