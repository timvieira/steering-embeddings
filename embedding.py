import numpy as np
import re
import scipy.spatial
import plotly.graph_objects as go
from numpy.linalg import norm, svd
from arsenal import Alphabet
from arsenal.iterextras import window


def mds(X, dimensions=2):
    """Multidimensional scaling: given a distance matrix, find low-dimensional
    points with similar interpoint distances."""
    E = -0.5 * X * X
    Er = np.array(np.mean(E, axis=1))
    Es = np.array(np.mean(E, axis=0))
    F = np.array(E - Er.T - Es + np.mean(E))
    U, S, _ = svd(F)
    Y = U * np.sqrt(S)
    return Y[:, :dimensions], S


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

    def _mds(self, words, dimensions=2):
        "Compute MDS coordinates and eigenvalues for a list of words."
        if self.dim <= dimensions:
            return np.array([self(w) for w in words]), None
        else:
            return mds(self.distance_matrix(words), dimensions=dimensions)

    @staticmethod
    def _add_arrow_3d(fig, start, end, color='rgba(0,0,255,0.4)', width=3, head_scale=0.04):
        "Add a 3D line with a mesh arrowhead at the endpoint."
        d = end - start
        length = norm(d)
        if length == 0:
            return
        d_hat = d / length
        # find two perpendicular vectors for the arrowhead base
        if abs(d_hat[0]) < 0.9:
            perp = np.cross(d_hat, np.array([1, 0, 0]))
        else:
            perp = np.cross(d_hat, np.array([0, 1, 0]))
        perp = perp / norm(perp)
        perp2 = np.cross(d_hat, perp)
        # arrowhead: a 4-sided pyramid
        h = head_scale                   # absolute arrowhead height
        r = h * 0.35                     # radius of base
        tip = end
        base_center = end - h * d_hat
        # 4 base points
        b = [base_center + r * (np.cos(a) * perp + np.sin(a) * perp2)
             for a in [0, np.pi/2, np.pi, 3*np.pi/2]]
        verts = np.array([tip] + b)      # 0=tip, 1-4=base
        # shaft stops at the arrowhead base
        fig.add_trace(go.Scatter3d(
            x=[start[0], base_center[0]], y=[start[1], base_center[1]],
            z=[start[2], base_center[2]],
            mode='lines', line=dict(color=color, width=width),
            showlegend=False,
        ))
        # arrowhead mesh: 4 side triangles + 2 base triangles
        fig.add_trace(go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=[0, 0, 0, 0, 1, 1],
            j=[1, 2, 3, 4, 2, 3],
            k=[2, 3, 4, 1, 3, 4],
            color=color, opacity=0.8,
            showlegend=False, hoverinfo='skip',
        ))

    @staticmethod
    def _variance_subtitle(S, dimensions):
        if S is None:
            return ''
        pct = S[:dimensions].sum() / S.sum() * 100
        return f'MDS: {dimensions}D captures {pct:.1f}% of variance'

    def plot_paths(self, G, connect_groups=False, dimensions=2):
        G = groups(G)
        words = Alphabet()
        words.add_many(y for x in G for y in x)
        Y, S = self._mds(words, dimensions=dimensions)

        fig = go.Figure()
        if dimensions == 3:
            fig.add_trace(go.Scatter3d(
                x=Y[:,0], y=Y[:,1], z=Y[:,2], mode='markers+text',
                text=list(words), textposition='top center',
                marker=dict(size=4, color='blue'),
            ))
            for p in G:
                for x, y in window(p, 2):
                    self._add_arrow_3d(fig, Y[words[x]], Y[words[y]])
            if connect_groups and len(G) > 1:
                n = min(len(p) for p in G)
                for i in range(n):
                    for g1, g2 in window(G, 2):
                        if i < len(g1) and i < len(g2):
                            p0, p1 = Y[words[g1[i]]], Y[words[g2[i]]]
                            fig.add_trace(go.Scatter3d(
                                x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                                mode='lines',
                                line=dict(color='rgba(255,0,0,0.3)', width=2, dash='dot'),
                                showlegend=False,
                            ))
        else:
            fig.add_trace(go.Scatter(
                x=Y[:,0], y=Y[:,1], mode='markers+text',
                text=list(words), textposition='top center',
                marker=dict(size=6, color='blue'),
            ))
            for p in G:
                for x, y in window(p, 2):
                    x0, y0 = Y[words[x]]
                    x1, y1 = Y[words[y]]
                    fig.add_annotation(
                        x=x1, y=y1, ax=x0, ay=y0,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1.5,
                        arrowwidth=1.5, arrowcolor='rgba(0,0,255,0.4)',
                    )
            if connect_groups and len(G) > 1:
                n = min(len(p) for p in G)
                for i in range(n):
                    for g1, g2 in window(G, 2):
                        if i < len(g1) and i < len(g2):
                            x0, y0 = Y[words[g1[i]]]
                            x1, y1 = Y[words[g2[i]]]
                            fig.add_trace(go.Scatter(
                                x=[x0, x1], y=[y0, y1], mode='lines',
                                line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dot'),
                                showlegend=False,
                            ))
        fig.update_layout(
            width=800, height=500, showlegend=False,
            title=self._variance_subtitle(S, dimensions),
        )
        fig.show()

    def plot_analogy(self, a, b, c, n, dimensions=2):
        ab = self._analogy(a, b, c, n=n)
        ba = self._analogy(b, a, c, n=n)
        words = Alphabet()
        words.add_many([a, b, c] + ab + list(set(ba) - set(ab)))
        Y, S = self._mds(words, dimensions=dimensions)

        fig = go.Figure()
        if dimensions == 3:
            fig.add_trace(go.Scatter3d(
                x=Y[:,0], y=Y[:,1], z=Y[:,2], mode='markers+text',
                text=list(words), textposition='top center',
                marker=dict(size=4, color='blue'),
            ))
            quad = Y[:4]
            fig.add_trace(go.Scatter3d(
                x=quad[:,0], y=quad[:,1], z=quad[:,2], mode='markers',
                marker=dict(size=8, color='red'),
            ))
            for i, j in [(0, 1), (2, 3)]:
                self._add_arrow_3d(fig, Y[i], Y[j], color='red', width=4)
        else:
            fig.add_trace(go.Scatter(
                x=Y[:,0], y=Y[:,1], mode='markers+text',
                text=list(words), textposition='top center',
                marker=dict(size=6, color='blue'),
            ))
            quad = Y[:4]
            fig.add_trace(go.Scatter(
                x=quad[:,0], y=quad[:,1], mode='markers',
                marker=dict(size=10, color='red'),
            ))
            for i, j in [(0, 1), (2, 3)]:
                fig.add_annotation(
                    x=Y[j,0], y=Y[j,1], ax=Y[i,0], ay=Y[i,1],
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.5,
                    arrowwidth=2, arrowcolor='red',
                )
        fig.update_layout(
            width=800, height=500, showlegend=False,
            title=self._variance_subtitle(S, dimensions),
        )
        fig.show()

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
    from path import Path

    if not Path('vecs.npz').exists():

        if not Path('data/glove.6B.100d.txt').exists():
            print('Download glove.6B.100d.zip from https://nlp.stanford.edu/projects/glove/')
            print('and unzip it')
            return

        with timeit('load'):
            glove_ix, glove_vecs = load_vecs('data/glove.6B.100d.txt')
        np.savez_compressed('vecs', vec = emb.vec, voc = np.array(list(emb.dom), dtype=str))

    with timeit('load-numpy'):
        with np.load('vecs.npz') as data:
            glove_vecs = data['vec']
            glove_ix = data['voc']

    emb = Embeddings(normalize_rows(glove_vecs), Alphabet(glove_ix))

    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('-i', '--interactive', action='store_true', help='open interactive shell')
    args = p.parse_args()
    if args.interactive:
        print()
        print('use object `emb`')
        from IPython import embed; embed()
        return

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
    Y, S = mds(D)
    pos = dict(zip(ww, Y))

    fig = go.Figure()
    xs = [pos[w][0] for w in words]
    ys = [pos[w][1] for w in words]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers+text',
        text=words, textposition='top center',
        marker=dict(size=6, color='blue'),
    ))
    for w in words:
        x0, y0 = pos[w]
        x1, y1 = pos['*'+w]
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowwidth=1.5, arrowcolor='rgba(0,0,255,0.4)',
        )
    pct = S[:2].sum() / S.sum() * 100
    fig.update_layout(
        width=800, height=500, showlegend=False,
        title=f'MDS: 2D captures {pct:.1f}% of variance',
    )
    fig.show()


if __name__ == '__main__':
    test()
