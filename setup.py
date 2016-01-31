from setuptools import setup, find_packages
import sys, os

version = '0.1'

long_description = "A numeric type represented as an integer coefficient and a base-10 exponent"

setup \
    ( name             = 'scientific'
    , version          = version
    , description      = "Port of the Haskell 'scientific' package"
    , long_description = long_description
    , classifiers      = []
    , keywords         = ''
    , author           = 'Chris Martin'
    , author_email     = 'ch.martin@gmail.com'
    , url              = ''
    , license          = 'WTFPL'
    , packages         = find_packages(exclude=['ez_setup', 'examples', 'tests'])
    )
