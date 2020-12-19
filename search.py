# The idea below is sort of nonsense: I wanted to generate an "interpretable 
# interpolation" between two concepts by traversing a graph.  The problem with the 
# specific execution is that the graph I created is defined in a silly way: if i 
# crank up the number of neighbor I will always prefer to go from start to end 
# in a single hop.

from arsenal.deathrow.prioritydict import prioritydict
from collections import defaultdict


def search(start, end, radius, **kw):
    y = emb(end)                        # target end point
    def h(w): return norm(emb(w) - y)   # search heuristic

    def edges(X):
        x = emb(X)
        for W in emb.query_radius(x, radius):
            #print(f'{X} -> {W}: {c} + {e}')
            yield W, norm(x - emb(W))

    def terminal(X):
        return X == end
    
    return run_search(start, edges, terminal, h, **kw)


def run_search(start, edges, terminal, h, max_pops=None):

    Q = prioritydict()
    C = defaultdict(lambda: np.inf)
    P = {}
    
    C[start] = 0
    P[start] = start
    Q[start] = h(start)

    N = 0
    while Q:
        X = Q.pop_smallest()
        N += 1
        
        if max_pops is not None and N > max_pops:
            break
        
#        print(X)
        if N % 25 == 0: print(f'explored {N} nodes, q size=', len(Q))
            
        c = C[X]
        
        if terminal(X):
            print('FINISHED!', start, '~~~>', X, 'cost=', c, 'path=', P[X])
            return c, P[X]

        for W, e in edges(X):
            if c + e < C[W]:
                C[W] = c + e
                Q[W] = C[W] + h(W)
                P[W] = (P[X], W)

    print('failed')
    return 

X, Y = "apple", "orange"
for r in [1, .9, .8, .7, .6, .5]:
    p = search(X, Y, r*norm(emb(X) - emb(Y)), max_pops=100)
    if p is None: break
