from collections import namedtuple

class Either(object): pass
class Left  (Either, namedtuple('Left',  'value')): pass
class Right (Either, namedTuple('Right', 'value')): pass

def isLeft  (either): return isinstance(Left,  either)
def isRight (either): return isinstance(Right, either)

def fold(either, leftF, rightF):
    return (leftF if isLeft(either) else rightR)(either.value)
