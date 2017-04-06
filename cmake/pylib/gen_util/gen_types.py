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


FLOAT_MAP = { FloatType.FLOAT : TypeDescriptor('s', 'float'),
    FloatType.DOUBLE : TypeDescriptor('d', 'double'),
    FloatType.FLOAT_COMPLEX : TypeDescriptor('c', 'FloatComplex'),
    FloatType.DOUBLE_COMPLEX : TypeDescriptor('z', 'DoubleComplex') }


arch_list = [ 'avx2', 'avx' ]

ARCH_MAP = {'common' : TypeDescriptor('HPTT_ARCH_COMMON', 'libhptt_common.so'),
    'avx' : TypeDescriptor('HPTT_ARCH_AVX', 'libhptt_avx.so'),
    'avx2' : TypeDescriptor('HPTT_ARCH_AVX2', 'libhptt_avx2.so') }
