#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import enum


class TypeDescriptor(object):
  def __init__(self, abbrev, full):
    self.abbrev = abbrev
    self.full = full


class FloatType(enum.Enum):
  FLOAT = 0
  DOUBLE = 1
  FLOAT_COMPLEX = 2
  DOUBLE_COMPLEX = 3


class CoefTrans(enum.Enum):
  USE_NONE = 0
  USE_ALPHA = 1
  USE_BETA = 2
  USE_BOTH = 3


FLOAT_MAP = { FloatType.FLOAT : TypeDescriptor('s', 'float'),
    FloatType.DOUBLE : TypeDescriptor('d', 'double'),
    FloatType.FLOAT_COMPLEX : TypeDescriptor('c', 'FloatComplex'),
    FloatType.DOUBLE_COMPLEX : TypeDescriptor('z', 'DoubleComplex') }


COEF_TRANS_MAP = { CoefTrans.USE_NONE : TypeDescriptor('none', 'USE_NONE'),
    CoefTrans.USE_ALPHA : TypeDescriptor('alpha', 'USE_ALPHA'),
    CoefTrans.USE_BETA : TypeDescriptor('beta', 'USE_BETA'),
    CoefTrans.USE_BOTH : TypeDescriptor('both', 'USE_BOTH') }
