from collections import namedtuple

class Maybe(object): pass
class Nothing(Maybe): pass
class Just(Maybe, namedtuple('Just', 'value')): pass

def isNothing (maybe): return isinstance(Nothing, maybe)
def isJust    (maybe): return isinstance(Just,    maybe)

def fromNone(x): return Nothing if x is None else Just(x)

def fold(maybe, ifNothing, ifJust):
    return ifNothing if isNothing(maybe) else ifJust(maybe.value)

def map(f, x): return Just(f(x.value)) if isJust(x) else Nothing
