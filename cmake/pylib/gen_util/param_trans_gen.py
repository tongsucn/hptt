#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gen_util.gen_types import (FloatType, FLOAT_MAP)


TARGET_PREFIX = 'parameter_trans'

class IncTarget(object):
  def __init__(self, **kwargs):
    dtypes = kwargs['dtype']
    orders = kwargs['order']
    suffix = kwargs['suffix']

    self.filename = ['%s_%s' % (TARGET_PREFIX, suffix)]
    temp_content = '''#pragma once
#ifndef HPTT_GEN_%s_GEN_TCC_
#define HPTT_GEN_%s_GEN_TCC_
''' % (TARGET_PREFIX.upper(), TARGET_PREFIX.upper())
    for dtype in dtypes:
      for order in orders:
        temp_content += '''
extern template struct ParamTrans<TensorWrapper<%s, %d>>;''' % (
    FLOAT_MAP[dtype].full, order)

    temp_content += '\n\n#endif'
    self.content = [temp_content]


class SrcTarget(object):
  def __init__(self, **kwargs):
    dtypes = kwargs['dtype']
    orders = kwargs['order']
    suffix = kwargs['suffix']

    self.filename = []
    self.content = []

    for dtype in dtypes:
      self.filename.append('%s_%s_%s' % (TARGET_PREFIX, FLOAT_MAP[dtype].abbrev,
          suffix))
      temp_content = '''#include <hptt/param/parameter_trans.h>

#include <hptt/types.h>
#include <hptt/tensor.h>

namespace hptt {
'''
      for order in orders:
        temp_content +='''
template struct ParamTrans<TensorWrapper<%s, %d>>;''' % (FLOAT_MAP[dtype].full,
    order)

      temp_content += '\n\n}'
      self.content.append(temp_content)
