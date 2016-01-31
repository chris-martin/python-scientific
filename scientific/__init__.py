"""
Scientific numbers are arbitrary precision. They are represented
using scientific notation. The implementation uses an integer
coefficient and a base-10 exponent.
"""

__all__ = \
    [ 'Scientific'

    # Construction
    , 'scientific'

    # Projections
    , 'coefficient'
    , 'base10Exponent'

    # Predicates
    , 'isFloating'
    , 'isInteger'

    # Conversions
    , 'fromRationalRepetend'
    , 'toRationalRepetend'
    , 'floatingOrInteger'
    , 'toRealFloat'
    , 'toBoundedRealFloat'
    , 'toBoundedInteger'
    , 'fromFloatDigits'

    # Pretty printing
    , 'formatScientific'
    , 'toDecimalDigits'

    # Normalization
    , 'normalize'
    ]

from collections import namedtuple

from . import either, map, maybe, rational
from .either import Left, Right, isLeft, isRight
from .maybe import Just, Nothing, isJust, isNothing
from .rational import Rational

#---------------------------------------------------------------------
#  Type
#---------------------------------------------------------------------


class Scientific(namedtuple('Scientific',
                            ['coefficient', 'base10Exponent'])):
    """
    An arbitrary-precision number represented using scientific notation.

    This type describes the set of all real numbers which have a finite
    decimal expansion.

    Attributes:

        coefficient :: long

            The coefficient of a scientific number.

            Note that this number is not necessarily normalized, i.e.
            it could contain trailing zeros.

            Scientific numbers are automatically normalized when pretty
            printed or in ``toDecimalDigits``.

            Use ``normalize`` to do manual normalization.


        base10Exponent :: long

            The base-10 exponent of a scientific number.
    """
    def __eq__(self, other): return toRational(self) == toRational(other)
    def __ne__(self, other): return toRational(self) != toRational(other)
    def __lt__(self, other): return toRational(self) <  toRational(other)
    def __le__(self, other): return toRational(self) <= toRational(other)
    def __gt__(self, other): return toRational(self) >  toRational(other)
    def __ge__(self, other): return toRational(self) >= toRational(other)

    def __cmp__(self, other):
        pass  # todo

    def __nonzero__(self):
        pass  # todo

    def __add__(self, other):
        (c1, e1) = self
        (c2, e2) = other
        if e1 < e2:
            l = magnitude (e2 - e1)
            return Scientific( c1   + c2*l, e1 )
        else:
            r = magnitude (e1 - e2)
            return Scientific( c1*r + c2  , e2 )

    def __sub__(self, other):
        (c1, e1) = self
        (c2, e2) = other
        if e1 < e2:
            l = magnitude (e2 - e1)
            return Scientific( c1   - c2*l, e1 )
        else:
            r = magnitude (e1 - e2)
            return Scientific( c1*r - c2  , e2 )

    def __mul__(self, other):
        (c1, e1) = self
        (c2, e2) = other
        return Scientific( c1 * c2, e1 + e2 )

    def __floordiv__(self, other):
        pass  # todo
    def __mod__(self, other):
        pass  # todo
    def __divmod__(self, other):
        pass  # todo
    def __pow__(self, other, modulo=None):
        pass  # todo
    def __lshift__(self, other):
        pass  # todo
    def __rshift__(self, other):
        pass  # todo
    def __and__(self, other):
        pass  # todo
    def __xor__(self, other):
        pass  # todo
    def __or__(self, other):
        pass  # todo

    def __div__(self, other):
        """
        WARNING: ``/`` will diverge (i.e. loop and consume all space)
        when its output is a repeating decimal.
        """
        fromRational(toRational(self) / toRational(other))

    def __truediv__(self, other):
        pass  # todo
    # The division operator (/) is implemented by these methods. The __truediv__() method is used when __future__.division is in effect, otherwise __div__() is used. If only one of these two methods is defined, the object will not support division in the alternate context; TypeError will be raised instead.

    def __neg__(self):
        (c, e) = self
        return Scientific( -c, e )

    def __pos__(self):
        pass  # todo

    def __abs__(self):
        (c, e) = self
        return Scientific( (abs(c)), e )

    def __invert__(self):
        pass  # todo
    # Called to implement the unary arithmetic operations (-, +, abs() and ~).

    def __complex__(self):
        pass  # todo
    def __int__(self):
        pass  # todo
    def __long__(self):
        pass  # todo
    def __float__(self):
        pass  # todo
    # Called to implement the built-in functions complex(), int(), long(), and float(). Should return a value of the appropriate type.

    def __oct__(self):
        pass  # todo
    def __hex__(self):
        pass  # todo
    # Called to implement the built-in functions oct() and hex(). Should return a string value.

    def __index__(self):
        pass  # todo
    # Called to implement operator.index(). Also called whenever Python needs an integer object (such as in slicing). Must return an integer (int or long).

    def __coerce__(self, other):
        pass  # todo
    # Called to implement “mixed-mode” numeric arithmetic. Should either return a 2-tuple containing self and other converted to a common numeric type, or None if conversion is impossible. When the common type would be the type of other, it is sufficient to return None, since the interpreter will also ask the other object to attempt a coercion (but sometimes, if the implementation of the other type cannot be changed, it is useful to do the conversion to the other type here). A return value of NotImplemented is equivalent to returning None.

def scientific(c, e):
    """
    Constructs a scientific number ``c * 10 ^ e``.
    :param c: The coefficient
    :param e: The base-10 exponent
    :return: Scientific
    """
    return Scientific(long(c), long(e))

def toRational(x):
    """
    WARNING: ``toRational`` needs to compute the magnitude ``10^e``.
    If applied to a huge exponent this could fill up all space
    and crash your program!

    Avoid applying ``toRational`` (or ``realToFrac``) to scientific numbers
    coming from an untrusted source and use ``toRealFloat`` instead. The
    latter guards against excessive space usage.

    :param x: Scientific
    :return: Rational
    """
    (c, e) = x
    if e < 0:
        return Rational( c, magnitude(-e) )
    else:
        return Rational( c * magnitude(e), 1 )

def recip(x):
    """
    Reciprocal fraction.

    WARNING: ``recip`` will diverge (i.e. loop and consume all space)
    when its output is a repeating decimal.

    :param x: Scientific
    :return: Scientific
    """
    return fromRational(rational.recip(toRational(x)))

def fromRational(rat):
    """
    ``fromRational`` will diverge when the input ``Rational`` is a repeating decimal.
    Consider using ``fromRationalRepetend`` for these rationals which will detect
    the repetition and indicate where it starts.

    :param x: Rational
    :return: Scientific
    """
    def longDiv(c, e, n):
        """
        Divide the numerator by the denominator using long division.

        :param c: long
        :param e: long
        :param n: long
        :return: Scientific
        """
        if n == 0:
            return Scientific(c, e)
        else:
            # TODO: Use a logarithm here!
            # TODO: Can't use tail recursion like this in python!
            if n < d:
                return longDiv(c * 10, e - 1, n * 10)
            else:
                (q, r) = quotRemInteger(n, d)
                return longDiv(c+q, e, r)

    d = rat.denominator

    if d == 0:
        raise ZeroDivisionError
    else:
        return rational.positivize(longDiv(0, 0), rat.numerator)

def fromRationalRepetend(rat, limit=Nothing):
    """
    Like ``fromRational``, this function converts a ``Rational`` to a ``Scientific``
    but instead of diverging (i.e loop and consume all space) on repeating decimals
    it detects the repeating part, the "repetend", and returns where it starts.

    To detect the repetition this function consumes space linear in the number of
    digits in the resulting scientific. In order to bound the space usage an
    optional limit can be specified. If the number of digits reaches this limit
    ``Left (s, r)`` will be returned. Here ``s`` is the ``Scientific`` constructed
    so far and ``r`` is the remaining ``Rational``. ``toRational s + r`` yields the
    original ``Rational``.

    If the limit is not reached or no limit was specified ``Right (s, mbRepetendIx)``
    will be returned. Here ``s`` is the ``Scientific`` without any repetition and
    ``mbRepetendIx`` specifies if and where in the fractional part the repetend begins.

    For example:

        fromRationalRepetend Nothing (1 % 28) == Right (3.571428e-2, Just 2)

    This represents the repeating decimal: ``0.03571428571428571428...``
    which is sometimes also unambiguously denoted as ``0.03(571428)``.
    Here the repetend is enclosed in parentheses and starts at the 3rd digit (index 2)
    in the fractional part. Specifying a limit results in the following:

        fromRationalRepetend (Just 4) (1 % 28) == Left (3.5e-2, 1 % 1400)

    You can expect the following property to hold.

        forall (mbLimit :: Maybe Int) (r :: Rational).
          r == (case fromRationalRepetend mbLimit r of
            Left (s, r') -> toRational s + r'
            Right (s, mbRepetendIx) ->
              case mbRepetendIx of
                Nothing         -> toRational s
                Just repetendIx -> toRationalRepetend s repetendIx)

    :param limit: Maybe long
    :param rat: Rational
    :return: Either (Scientific, Rational) (Scientific, Maybe Int)
    """

    def longDiv(n):
        """
        :param n: long
        :return: Either (Scientific, Rational) (Scientific, Maybe Int)
        """
        return limit.fold(longDivNoLimit(0, 0, map.empty, n),
                          lambda l: longDivWithLimit(-l, n))

    # todo - this tail recursion won't stand, man
    def longDivNoLimit(c, e, ns, n):
        """
        Divide the numerator by the denominator using long division.

        :param c: long
        :param e: long
        :param ns: Map long long
        :param n: long
        :return (Scientific, Maybe long)
        """
        if n == 0:
            return (Scientific(c, e), Nothing)
        else:
            e_prime = map.lookup(n, ns)
            if isJust(e_prime):
                return (Scientific(c, e), maybe.map(negate, e_prime))
            elif n < rat.denominator:
                return longDivNoLimit(c * 10, e - 1, map.insert(n, e, ns), n * 10)
            else:
                (q, r) = quotRemInteger(n, rat.denominator)
                return longDivNoLimit(c + q, e, ns, r)

    def longDivWithLimit(l, n):
        """
        :param l: long
        :param n: long
        :return: Either (Scientific, Rational) (Scientific, Maybe Int)
        """

        # todo - this tail recursion won't stand, man
        def go(c, e, ns, n):
            """
            :param c: long
            :param e: long
            :param ns: Map long long
            :param n: long
            :return: Either (Scientific, Rational) (Scientific, Maybe Int)
            """
            if n == 0:
                return Right(Scientific(c, e), Nothing)
            else:
                e_prime = map.lookup(n, ns)
                if isJust(e_prime):
                    return Right(Scientific(c, e), maybe.map(negatve, e_prime))
                elif e <= l:
                    return Left(Scientific(c, e), n % (d * magnitude (-e)))
                elif n < d:
                    return go(c * 10, e - 1, map.insert(n, e, ns), n * 10)
                else:
                    (q, r) = quotRemInteger(n, rat.denominator)
                    return go(c + q, e, ns, r)

        return go(0, 0, map.empty)

    if rat.denominator == 0:
        raise ZeroDivisionError

    elif rat.numerator < 0:
        return either.fold(
            longDiv(-num),
            lambda (s, r ): Left(  (-s, -r) ),
            lambda (s, mb): Right( (-s, mb) ))

    else:
        return longDiv(num)

def toRationalRepetend(s, r):
    """
    Converts a ``Scientific`` with a "repetend" (a repeating part in the fraction),
    which starts at the given index, into its corresponding ``Rational``.

    For example to convert the repeating decimal ``0.03(571428)`` you would use:

        toRationalRepetend(0.03571428, 2) == 1 % 28

    Preconditions for ``toRationalRepetend s r``:

        * @r >= 0@

        * @r < -(base10Exponent s)@

    Also see: ``fromRationalRepetend``.

    :param s: Scientific
    :param r: long - Repetend index
    :return Rational:
    """
    if r < 0:
        raise ValueError("toRationalRepetend: Negative repetend index!")
    elif r >= f:
        raise ValueError("toRationalRepetend: Repetend index >= than number of digits in the fractional part!")
    else:
        c  = coefficient(s)
        e  = base10Exponent(s)
        f = -e        # Size of the fractional part.
        n = f - r     # Size of the repetend.
        m = magnitude(n)
        (nonRepetend, repetend) = quotRemInteger(c, m)
        nines = m - 1
        return fromInteger(nonRepetend + (repetend % nines)) / fromInteger(magnitude(r))

def properFraction(s):
    """
    Takes a Scientific number ``s`` and returns a pair ``(n,f)`` such that ``s = n+f``, and:

        * ``n`` is an integral number with the same sign as ``s``; and

        * ``f`` is a fraction with the same type and sign as ``s``,
          and with absolute value less than ``1``.

    :param s:
    :return: (long, Scientific)
    """
    (c, e) = s
    if e < 0:
        if dangerouslySmall(c, e):
            return (0, s)
        else:
            (q, r) = quotRemInteger(c, magnitude(-1))
            return (fromInteger(q), Scientific(r, e))
    else:
        return (toIntegral(s), 0)

def truncate(s):
    """
    The integer nearest ``s`` between zero and ``s``.

    :param s: Scientific
    :return: long
    """
    truncate = whenFloating $ \c e ->
                 if dangerouslySmall c e
                 then 0
                 else fromInteger $ c `quotInteger` magnitude (-e)

def round(s):
    """
    The nearest integer to ``s``; the even integer if ``s`` is equidistant between two integers.
    """
    round = whenFloating $ \c e ->
              if dangerouslySmall c e
              then 0
              else let (#q, r#) = c `quotRemInteger` magnitude (-e)
                       n = fromInteger q
                       m | r < 0     = n - 1
                         | otherwise = n + 1
                       f = Scientific r e
                   in case signum $ coefficient $ abs f - 0.5 of
                        -1 -> n
                        0  -> if even n then n else m
                        1  -> m
                        _  -> error "round default defn: Bad value"

def ceiling(s):
    """
    The least integer not less than ``s``.
    """
    ceiling = whenFloating $ \c e ->
                if dangerouslySmall c e
                then if c <= 0
                     then 0
                     else 1
                else case c `quotRemInteger` magnitude (-e) of
                       (#q, r#) | r <= 0    -> fromInteger q
                                | otherwise -> fromInteger (q + 1)


def floor(s):
    """
    The greatest integer not greater than ``s``.
    """
    floor = whenFloating $ \c e ->
              if dangerouslySmall c e
              then if c < 0
                   then -1
                   else 0
              else fromInteger (c `divInteger` magnitude (-e))


#---------------------------------------------------------------------
#  Internal utilities
#---------------------------------------------------------------------

def dangerouslySmall(c, e):
    """
    This function is used in the ``RealFrac`` methods to guard against
    computing a huge magnitude (-e) which could take up all space.

    Think about parsing a scientific number from an untrusted
    string. An attacker could supply ``1e-1000000000``. Lets say we want to
    ``floor`` that number to an ``Int``. When we naively try to floor it
    using:

        floor = whenFloating $ \c e ->
                  fromInteger (c `div` magnitude (-e))

    We will compute the huge Integer: ``magnitude 1000000000``. This
    computation will quickly fill up all space and crash the program.

    Note that for large *positive* exponents there is no risk of a
    space-leak since ``whenFloating`` will compute:

        fromInteger c * magnitude e :: a

    where ``a`` is the target type (Int in this example). So here the
    space usage is bounded by the target type.

    For large negative exponents we check if the exponent is smaller
    than some limit (currently -324). In that case we know that the
    scientific number is really small (unless the coefficient has many
    digits) so we can immediately return -1 for negative scientific
    numbers or 0 for positive numbers.

    More precisely if ``dangerouslySmall c e`` returns ``True`` the
    scientific number ``s`` is guaranteed to be between:
    ``-0.1 > s < 0.1``.

    Note that we avoid computing the number of decimal digits in c
    (log10 c) if the exponent is not below the limit.

    :param c: long
    :param e: long
    :return: bool
    """
    return e < -limit and e < (-integerLog10(abs(c))) - 1

limit = maxExpt  # long

def positivize(f, x):
    """
    positivize :: (Ord a, Num a, Num b) => (a -> b) -> (a -> b)
    """
    return -(f(-x)) if x < 0 else f(x)

def whenFloating(f, s):
    """
    whenFloating :: (Num a) => (Integer -> Int -> a) -> Scientific -> a
    """
    (c, e) = s
    return f(c, e) if e < 0 else toIntegral(s)

def toIntegral(s):
    """
    toIntegral :: (Num a) => Scientific -> a

    Precondition: the ``Scientific`` ``s`` needs to be an integer:
    ``base10Exponent (normalize s) >= 0``
    """
    (c, e) = s
    return fromInteger(c) * magnitude(e)


#---------------------------------------------------------------------
#  Exponentiation with a cache for the most common numbers.
#---------------------------------------------------------------------

maxExpt = 324  # long - The same limit as in GHC.Float.

expts10 :: V.Vector Integer
expts10 = runST $ do
    mv <- VM.unsafeNew maxExpt
    VM.unsafeWrite mv 0  1
    VM.unsafeWrite mv 1 10
    let go !ix
          | ix == maxExpt = V.unsafeFreeze mv
          | otherwise = do
              VM.unsafeWrite mv  ix        xx
              VM.unsafeWrite mv (ix+1) (10*xx)
              go (ix+2)
          where
            xx = x * x
            x  = V.unsafeIndex expts10 half
#if MIN_VERSION_base(4,5,0)
            !half = ix `unsafeShiftR` 1
#else
            !half = ix `shiftR` 1
#endif
    go 2


def magnitude(e):
    """
    magnitude :: (Num a) => Int -> a

    magnitude e == 10 ^ e
    """
    def cachedPow10(p): return fromInteger(V.unsafeIndex(expts10, p))

    if e < maxExpt:
        return cachedPow10(e)
    else:
        hi = maxExpt - 1
        return cachedPow10(hi * 10 ^ (e - hi))


#---------------------------------------------------------------------
#  Conversions
#---------------------------------------------------------------------

-- | Convert a 'RealFloat' (like a 'Double' or 'Float') into a 'Scientific'
-- number.
--
-- Note that this function uses 'Numeric.floatToDigits' to compute the digits
-- and exponent of the 'RealFloat' number. Be aware that the algorithm used in
-- 'Numeric.floatToDigits' doesn't work as expected for some numbers, e.g. as
-- the 'Double' @1e23@ is converted to @9.9999999999999991611392e22@, and that
-- value is shown as @9.999999999999999e22@ rather than the shorter @1e23@; the
-- algorithm doesn't take the rounding direction for values exactly half-way
-- between two adjacent representable values into account, so if you have a
-- value with a short decimal representation exactly half-way between two
-- adjacent representable values, like @5^23*2^e@ for @e@ close to 23, the
-- algorithm doesn't know in which direction the short decimal representation
-- would be rounded and computes more digits
fromFloatDigits :: (RealFloat a) => a -> Scientific
fromFloatDigits = positivize fromPositiveRealFloat
    where
      fromPositiveRealFloat r = go digits 0 0
        where
          (digits, e) = Numeric.floatToDigits 10 r

          go []     !c !n = Scientific c (e - n)
          go (d:ds) !c !n = go ds (c * 10 + fromIntegral d) (n + 1)

-- | Safely convert a 'Scientific' number into a 'RealFloat' (like a 'Double' or a
-- 'Float').
--
-- Note that this function uses 'realToFrac' (@'fromRational' . 'toRational'@)
-- internally but it guards against computing huge Integer magnitudes (@10^e@)
-- that could fill up all space and crash your program. If the 'base10Exponent'
-- of the given 'Scientific' is too big or too small to be represented in the
-- target type, Infinity or 0 will be returned respectively. Use
-- 'toBoundedRealFloat' which explicitly handles this case by returning 'Left'.
--
-- Always prefer 'toRealFloat' over 'realToFrac' when converting from scientific
-- numbers coming from an untrusted source.
toRealFloat :: (RealFloat a) => Scientific -> a
toRealFloat = either id id . toBoundedRealFloat

-- | Preciser version of `toRealFloat`. If the 'base10Exponent' of the given
-- 'Scientific' is too big or too small to be represented in the target type,
-- Infinity or 0 will be returned as 'Left'.
toBoundedRealFloat :: forall a. (RealFloat a) => Scientific -> Either a a
toBoundedRealFloat s@(Scientific c e)
    | c == 0                                       = Right 0
    | e >  limit && e > hiLimit                    = Left  $ sign (1/0) -- Infinity
    | e < -limit && e < loLimit && e + d < loLimit = Left  $ sign 0
    | otherwise                                    = Right $ realToFrac s
  where
    (loLimit, hiLimit) = exponentLimits (undefined :: a)

    d = integerLog10' (abs c)

    sign x | c < 0     = -x
           | otherwise =  x

exponentLimits :: forall a. (RealFloat a) => a -> (Int, Int)
exponentLimits _ = (loLimit, hiLimit)
    where
      loLimit = floor   (fromIntegral lo     * log10Radix) -
                ceiling (fromIntegral digits * log10Radix)
      hiLimit = ceiling (fromIntegral hi     * log10Radix)

      log10Radix :: Double
      log10Radix = logBase 10 $ fromInteger radix

      radix    = floatRadix  (undefined :: a)
      digits   = floatDigits (undefined :: a)
      (lo, hi) = floatRange  (undefined :: a)

-- | Convert a `Scientific` to a bounded integer.
--
-- If the given `Scientific` doesn't fit in the target representation, it will
-- return `Nothing`.
--
-- This function also guards against computing huge Integer magnitudes (@10^e@)
-- that could fill up all space and crash your program.
toBoundedInteger :: forall i. (Integral i, Bounded i) => Scientific -> Maybe i
toBoundedInteger s
    | c == 0    = fromIntegerBounded 0
    | integral  = if dangerouslyBig
                  then Nothing
                  else fromIntegerBounded n
    | otherwise = Nothing
  where
    c = coefficient s

    integral = e >= 0 || e' >= 0

    e  = base10Exponent s
    e' = base10Exponent s'

    s' = normalize s

    dangerouslyBig = e > limit &&
                     e > integerLog10' (max (abs iMinBound) (abs iMaxBound))

    fromIntegerBounded :: Integer -> Maybe i
    fromIntegerBounded i
        | i < iMinBound || i > iMaxBound = Nothing
        | otherwise                      = Just $ fromInteger i

    iMinBound = toInteger (minBound :: i)
    iMaxBound = toInteger (maxBound :: i)

    -- This should not be evaluated if the given Scientific is dangerouslyBig
    -- since it could consume all space and crash the process:
    n :: Integer
    n = toIntegral s'

-- | @floatingOrInteger@ determines if the scientific is floating point
-- or integer. In case it's floating-point the scientific is converted
-- to the desired 'RealFloat' using 'toRealFloat'.
--
-- Also see: 'isFloating' or 'isInteger'.
floatingOrInteger :: (RealFloat r, Integral i) => Scientific -> Either r i
floatingOrInteger s
    | base10Exponent s  >= 0 = Right (toIntegral   s)
    | base10Exponent s' >= 0 = Right (toIntegral   s')
    | otherwise              = Left  (toRealFloat  s')
  where
    s' = normalize s


----------------------------------------------------------------------
-- Predicates
----------------------------------------------------------------------

-- | Return 'True' if the scientific is a floating point, 'False' otherwise.
--
-- Also see: 'floatingOrInteger'.
isFloating :: Scientific -> Bool
isFloating = not . isInteger

-- | Return 'True' if the scientific is an integer, 'False' otherwise.
--
-- Also see: 'floatingOrInteger'.
isInteger :: Scientific -> Bool
isInteger s = base10Exponent s  >= 0 ||
              base10Exponent s' >= 0
  where
    s' = normalize s


----------------------------------------------------------------------
-- Parsing
----------------------------------------------------------------------

instance Read Scientific where
    readPrec = Read.parens $ ReadPrec.lift (ReadP.skipSpaces >> scientificP)

-- A strict pair
data SP = SP !Integer {-# UNPACK #-}!Int

scientificP :: ReadP Scientific
scientificP = do
  let positive = (('+' ==) <$> ReadP.satisfy isSign) `mplus` return True
  pos <- positive

  let step :: Num a => a -> Int -> a
      step a digit = a * 10 + fromIntegral digit
      {-# INLINE step #-}

  n <- foldDigits step 0

  let s = SP n 0
      fractional = foldDigits (\(SP a e) digit ->
                                 SP (step a digit) (e-1)) s

  SP coeff expnt <- (ReadP.satisfy (== '.') >> fractional)
                    ReadP.<++ return s

  let signedCoeff | pos       =   coeff
                  | otherwise = (-coeff)

      eP = do posE <- positive
              e <- foldDigits step 0
              if posE
                then return   e
                else return (-e)

  (ReadP.satisfy isE >>
           ((Scientific signedCoeff . (expnt +)) <$> eP)) `mplus`
     return (Scientific signedCoeff    expnt)


foldDigits :: (a -> Int -> a) -> a -> ReadP a
foldDigits f z = do
    c <- ReadP.satisfy isDecimal
    let digit = ord c - 48
        a = f z digit

    ReadP.look >>= go a
  where
    go !a [] = return a
    go !a (c:cs)
        | isDecimal c = do
            _ <- ReadP.get
            let digit = ord c - 48
            go (f a digit) cs
        | otherwise = return a

isDecimal :: Char -> Bool
isDecimal c = c >= '0' && c <= '9'
{-# INLINE isDecimal #-}

isSign :: Char -> Bool
isSign c = c == '-' || c == '+'
{-# INLINE isSign #-}

isE :: Char -> Bool
isE c = c == 'e' || c == 'E'
{-# INLINE isE #-}


----------------------------------------------------------------------
-- Pretty Printing
----------------------------------------------------------------------

instance Show Scientific where
    show s | coefficient s < 0 = '-':showPositive (-s)
           | otherwise         =     showPositive   s
      where
        showPositive :: Scientific -> String
        showPositive = fmtAsGeneric . toDecimalDigits

        fmtAsGeneric :: ([Int], Int) -> String
        fmtAsGeneric x@(_is, e)
            | e < 0 || e > 7 = fmtAsExponent x
            | otherwise      = fmtAsFixed    x

fmtAsExponent :: ([Int], Int) -> String
fmtAsExponent (is, e) =
    case ds of
      "0"     -> "0.0e0"
      [d]     -> d : '.' :'0' : 'e' : show_e'
      (d:ds') -> d : '.' : ds' ++ ('e' : show_e')
      []      -> error "formatScientific/doFmt/FFExponent: []"
  where
    show_e' = show (e-1)

    ds = map intToDigit is

fmtAsFixed :: ([Int], Int) -> String
fmtAsFixed (is, e)
    | e <= 0    = '0':'.':(replicate (-e) '0' ++ ds)
    | otherwise =
        let
           f 0 s    rs  = mk0 (reverse s) ++ '.':mk0 rs
           f n s    ""  = f (n-1) ('0':s) ""
           f n s (r:rs) = f (n-1) (r:s) rs
        in
           f e "" ds
  where
    mk0 "" = "0"
    mk0 ls = ls

    ds = map intToDigit is

-- | Like 'show' but provides rendering options.
formatScientific :: FPFormat
                 -> Maybe Int  -- ^ Number of decimal places to render.
                 -> Scientific
                 -> String
formatScientific format mbDecs s
    | coefficient s < 0 = '-':formatPositiveScientific (-s)
    | otherwise         =     formatPositiveScientific   s
  where
    formatPositiveScientific :: Scientific -> String
    formatPositiveScientific s' = case format of
        Generic  -> fmtAsGeneric        $ toDecimalDigits s'
        Exponent -> fmtAsExponentMbDecs $ toDecimalDigits s'
        Fixed    -> fmtAsFixedMbDecs    $ toDecimalDigits s'

    fmtAsGeneric :: ([Int], Int) -> String
    fmtAsGeneric x@(_is, e)
        | e < 0 || e > 7 = fmtAsExponentMbDecs x
        | otherwise      = fmtAsFixedMbDecs x

    fmtAsExponentMbDecs :: ([Int], Int) -> String
    fmtAsExponentMbDecs x = case mbDecs of
                              Nothing  -> fmtAsExponent x
                              Just dec -> fmtAsExponentDecs dec x

    fmtAsFixedMbDecs :: ([Int], Int) -> String
    fmtAsFixedMbDecs x = case mbDecs of
                           Nothing  -> fmtAsFixed x
                           Just dec -> fmtAsFixedDecs dec x

    fmtAsExponentDecs :: Int -> ([Int], Int) -> String
    fmtAsExponentDecs dec (is, e) =
        let dec' = max dec 1 in
            case is of
             [0] -> '0' :'.' : take dec' (repeat '0') ++ "e0"
             _ ->
              let
               (ei,is') = roundTo (dec'+1) is
               (d:ds') = map intToDigit (if ei > 0 then init is' else is')
              in
              d:'.':ds' ++ 'e':show (e-1+ei)

    fmtAsFixedDecs :: Int -> ([Int], Int) -> String
    fmtAsFixedDecs dec (is, e) =
        let dec' = max dec 0 in
        if e >= 0 then
         let
          (ei,is') = roundTo (dec' + e) is
          (ls,rs)  = splitAt (e+ei) (map intToDigit is')
         in
         mk0 ls ++ (if null rs then "" else '.':rs)
        else
         let
          (ei,is') = roundTo dec' (replicate (-e) 0 ++ is)
          d:ds' = map intToDigit (if ei > 0 then is' else 0:is')
         in
         d : (if null ds' then "" else '.':ds')
      where
        mk0 ls = case ls of { "" -> "0" ; _ -> ls}

----------------------------------------------------------------------

-- | Similar to 'Numeric.floatToDigits', @toDecimalDigits@ takes a
-- positive 'Scientific' number, and returns a list of digits and
-- a base-10 exponent. In particular, if @x>=0@, and
--
-- > toDecimalDigits x = ([d1,d2,...,dn], e)
--
-- then
--
--     1. @n >= 1@
--     2. @x = 0.d1d2...dn * (10^^e)@
--     3. @0 <= di <= 9@
--     4. @null $ takeWhile (==0) $ reverse [d1,d2,...,dn]@
--
-- The last property means that the coefficient will be normalized, i.e. doesn't
-- contain trailing zeros.
toDecimalDigits :: Scientific -> ([Int], Int)
toDecimalDigits (Scientific 0  _)  = ([0], 1)
toDecimalDigits (Scientific c' e') =
    case normalizePositive c' e' of
      Scientific c e -> go c 0 []
        where
          go :: Integer -> Int -> [Int] -> ([Int], Int)
          go 0 !n ds = (ds, ne) where !ne = n + e
          go i !n ds = case i `quotRemInteger` 10 of
                         (# q, r #) -> go q (n+1) (d:ds)
                           where
                             !d = fromIntegral r


----------------------------------------------------------------------
-- Normalization
----------------------------------------------------------------------

-- | Normalize a scientific number by dividing out powers of 10 from the
-- 'coefficient' and incrementing the 'base10Exponent' each time.
--
-- You should rarely have a need for this function since scientific numbers are
-- automatically normalized when pretty-printed and in 'toDecimalDigits'.
normalize :: Scientific -> Scientific
normalize (Scientific c e)
    | c > 0 =   normalizePositive   c  e
    | c < 0 = -(normalizePositive (-c) e)
    | otherwise {- c == 0 -} = Scientific 0 0

normalizePositive :: Integer -> Int -> Scientific
normalizePositive !c !e = case quotRemInteger c 10 of
                            (# c', r #)
                                | r == 0    -> normalizePositive c' (e+1)
                                | otherwise -> Scientific c e

def negate(x): return -x
