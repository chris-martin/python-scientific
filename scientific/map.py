from .. import maybe

empty = {}

def insert(k, v, m):
    m_prime = dict(m)
    m_prime[k] = v
    return m_prime

def lookup(k, m): return maybe.fromNone(ns.get(n))
