import pylab as pl
import numpy as np
import re
import scipy.spatial
from numpy.linalg import norm, svd
from arsenal.viz.mds import mds
from arsenal import Alphabet
from arsenal.iterextras import window


#def cosine(a, b):
#    return (a @ b) / (np.linalg.norm(a)*np.linalg.norm(b))

#def least_similar(emb, w):
#    wv = emb(w)
#    return list(sorted([(cosine(wv, emb.vec[i]), emb.dom.lookup(i))
#                        for i in range(emb.vec.shape[0])]))


class Embeddings:
    def __init__(self, vec, dom):
        assert isinstance(dom, Alphabet)
        self.vec = vec
        self.dom = dom
        self.dim = vec.shape[1]
        assert len(dom) == vec.shape[0]
        self.kd = scipy.spatial.cKDTree(vec)

    def __call__(self, w):
        return self.vec[self.dom[w]]

    def most_similar(self, x, n, exclude=None):
        "n is the number we want to find. `exclude` are the words we cannot pick"
        _, inds = self.kd.query(x, n+len(exclude or []))
        if isinstance(inds, int): inds = [inds]
        top = self.dom.decode_many(inds)
        if exclude: top = [x for x in top if x not in exclude]
        return top[:n]

    def query_radius(self, x, radius):
        return self.dom.decode_many(self.kd.query_ball_point(x, radius))

    def analogy(self, x, n=1):
        [(a,b,c)] = re.findall('(\S+)\s*::\s*(\S+)\s*->\s*(\S+)', x)
        print(f'{x} :: [{", ".join(self._analogy(a, c, b, n=n))}]')

    def _analogy(self, a, b, c, n):
        "Top n results for a :: c -> b :: ?"
        return self.most_similar(self(b) - self(a) + self(c), n=n, exclude=[a, b, c])

    def distance_matrix(self, words):
        words = list(words)
        n = len(words)
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(0, i):
                D[i,j] = D[j,i] = norm(self(words[i]) - self(words[j]))
        return D

    def _plot_paths(self, paths):
        words = Alphabet()
        words.add_many(y for x in paths for y in x)

        if self.dim == 2:    # visualize 2d embeddings directly
            Y = np.array([self(w) for w in words])
        else:
            Y, _ = mds(self.distance_matrix(words))

        pl.figure(figsize=(12, 6))
        pl.scatter(Y[:,0], Y[:,1], s=20, c='b', marker='o')
        for label, x, y in zip(words, Y[:,0], Y[:,1]):
            pl.text(x, y, label, fontsize=12)
        pl.grid(True)

        return words, Y

    def plot_paths(self, G):
        G = groups(G)
        ix, pos = self._plot_paths(G)
        for p in G:
            for x,y in window(p, 2):
                xs, ys = np.array([pos[ix[x]], pos[ix[y]]]).T
                pl.arrow(xs[0], ys[0], (xs[1] - xs[0]), (ys[1] - ys[0]),
                         **arrow_style)

    def plot_analogy(self, a, b, c, n):
        ab = self._analogy(a, b, c, n=n)
        # look at the top words in the reverse direction to get words that
        # contrast when the relationship is asymmetric
        ba = self._analogy(b, a, c, n=n)
        words = [a,b,c] + ab + list(set(ba) - set(ab))
        _, pos = self._plot_paths([words])

        xs, ys = pos[:4].T
        pl.scatter(xs, ys, c='r')

        pl.arrow(xs[0], ys[0], (xs[1] - xs[0]), (ys[1] - ys[0]),
                 color='r', **arrow_style)
        pl.arrow(xs[2], ys[2], (xs[3] - xs[2]), (ys[3] - ys[2]),
                 color='r', **arrow_style)

    def subspace(self, G, K):
        """
        Find the K-dimensional subspace that best explains
        the variation within groups.

        The output is a K x D matrix, B.

        B @ v projects into K space,
        B @ v @ B projects back to D space.

        """
        G = groups(G)
        C = np.zeros((self.dim, self.dim))
        for g in G:
            C += np.cov(self.vec[self.dom.encode_many(g)].T)
        return svd(C)[0][:,:K].T

    def debias(self, G, K):
        # Apply the tranform to all of the embeddings and subtract it away.
        B = self.subspace(G, K)
        A = self.vec - self.vec @ B.T @ B
        A = normalize_rows(A)     # re-normalize embeddings
        z = self.__class__(A, self.dom)
        z.B = B
        return z


arrow_style = dict(length_includes_head=True, alpha=0.5, width=.005, lw=0,
                   head_width=0.03, head_length=.03)


def normalize_rows(A):
    return (A.T / norm(A, axis=1)).T     # re-normalize embeddings


def load_vecs(path):
    """ Loads in word vectors from path.
    Will return a dictionary of word to index, and a matrix of vectors (each word is a row)
    """
    vs = []; ws = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split()
            ws.append(line[0])
            vs.append(line[1:])
    vs = np.array(vs, dtype=float)
    alphabet = Alphabet()
    alphabet.add_many(ws)
    alphabet.freeze()
    print(f'Read in {len(ws)} words, dimensionality {vs.shape[1]}')
    return alphabet, vs


def groups(s):
    if not isinstance(s, str): return s
    g = []
    for x in s.split('\n'):
        x = re.sub('#.*', '', x)   # remove comments.
        x = x.strip()
        if x:
            g.append(x.split())
    return g


def test():
    from arsenal import timers, timeit

    with timeit('load-numpy'):
        with np.load('vecs.npz') as data:
            glove_vecs = data['vec']
            glove_ix = data['voc']

    #with timeit('load'):
    #    glove_ix, glove_vecs = load_vecs('data/glove.6B.100d.txt')
    #np.savez_compressed('vecs', vec = emb.vec, voc = np.array(list(emb.dom), dtype=str))

    emb = Embeddings(normalize_rows(glove_vecs), Alphabet(glove_ix))

    from IPython import embed; embed()

    gendered = """
    man   woman
    king  queen
    actor actress
    boy   girl
    dad   mom
    father mother
    john mary
    brother sister
    uncle aunt
    heir heiress
    duke duchess
    nephew niece
    sir madame
    # count countess
    male female
    masculine feminine
    """

    if 0:
        # MDS plot
        emb.plot_paths(gendered)

        # Subspace plot - these plots don't look great because they are not
        # capturing the other types of distance; thus the vectors clump together.
        B = emb.subspace(gendered, K=2)
        Embeddings(emb.vec @ B.T, emb.dom).plot_paths(gendered)

        # Debiased plot
        emb.debias(gendered, K=2).plot_paths(gendered)

    deb = emb.debias(gendered, K=10)

    professions = [
        'caretaker', 'homemaker',
        'doctor', 'nurse', 'programmer', 'teacher',
        'wife', 'husband', 'soldier', 'salesperson', 'analyst', 'therapist',
        'trainer', 'instructor', 'ceo', 'assistant', 'telemarketer',
        'bartender', 'clerk', 'designer', 'father', 'mother', 'scientist',
        'manager', 'boss', 'self-employed', 'employee',
    ]

    p = sorted(professions, key = lambda w: -norm(deb(w) - emb(w)))
    print(p)


    plot_change(professions, emb, deb)

    plot_change(gendered.strip().split(), emb, deb)

    pl.show()

    return

    def most_similar_slow(emb, x, n, exclude=None):
        vocab = list(emb.dom.keys())
        #vocab.sort(key = lambda y: norm(x - self(y)))
        vocab.sort(key = lambda y: -(x @ emb(y)))
        if exclude: vocab = [x for x in vocab if x not in exclude]
        return vocab[:n]


    T = timers()
    K = 5
    for x in ['man', 'cowboy', 'president']:
        with T['kd']:
            a = emb.most_similar(emb(x), n = K)
        with T['sort']:
            b = most_similar_slow(emb, emb(x), n = K)
        print()
        print(x)
        print(a)
        print(b)
    print()
    T.compare()


def plot_change(words, emb, deb):

    def distance_matrix(words):
        words = list(words)
        n = len(words)
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(0, i):

                u = words[i]
                v = words[j]

                u = deb(u[1:]) if u.startswith('*') else emb(u)
                v = deb(v[1:]) if v.startswith('*') else emb(v)

                D[i,j] = D[j,i] = norm(u - v)
        return D

    ww = words + ['*'+w for w in words]
    D = distance_matrix(ww)

    Y, _ = mds(D)

    pl.figure(figsize=(12, 6))
    pl.scatter(Y[:,0], Y[:,1], s=20, c='b', marker='o')

    pos = dict(zip(ww, Y))

    for w in words:
        x,y = pos[w]
        pl.text(x, y, w, fontsize=12)

        x1, y1 = pos['*'+w]

        pl.arrow(x, y, (x1 - x), (y1 - y), **arrow_style)

    pl.grid(True)


if __name__ == '__main__':
    test()
