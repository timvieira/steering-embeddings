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


def export(words, vecs, outpath, n=None):
    if n is not None:
        words = words[:n]
        vecs = vecs[:n]
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

    DATA_DIR.mkdir(exist_ok=True)

    export(words, vecs, DATA_DIR / 'glove-small.bin', n=10_000)
    export(words, vecs, DATA_DIR / 'glove-medium.bin', n=50_000)
    export(words, vecs, DATA_DIR / 'glove-large.bin', n=len(words))


if __name__ == '__main__':
    main()
