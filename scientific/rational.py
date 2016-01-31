from collections import namedtuple

class Rational(namedtuple('Rational', ['numerator', 'denominator'])):
    """
    ``a/b``, where ``a`` and ``b`` are of type ``long``.
    """
