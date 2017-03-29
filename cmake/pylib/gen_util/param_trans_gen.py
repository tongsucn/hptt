#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gen_util.gen_types import (FloatType, FLOAT_MAP)


TARGET_PREFIX = 'parameter_trans'

class IncTarget(object):
  def __init__(self, **kwargs):
    dtypes = kwargs['dtype']
    orders = kwargs['order']

    self.filename = ['%s_gen.tcc' % TARGET_PREFIX]
    temp_content = '''#pragma once
#ifndef HPTC_GEN_%s_GEN_TCC_
#define HPTC_GEN_%s_GEN_TCC_
''' % (TARGET_PREFIX.upper(), TARGET_PREFIX.upper())
    for dtype in dtypes:
      for order in orders:
        temp_content += '''
extern template class ParamTrans<
    TensorWrapper<%s, %d, MemLayout::COL_MAJOR>>;
extern template class ParamTrans<
    TensorWrapper<%s, %d, MemLayout::ROW_MAJOR>>;''' % (FLOAT_MAP[dtype].full,
    order, FLOAT_MAP[dtype].full, order)

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
      temp_content = '''#include <hptc/param/parameter_trans.h>

#include <hptc/types.h>
#include <hptc/compat.h>
#include <hptc/tensor.h>

namespace hptc {
'''
      for order in orders:
        temp_content +='''
template struct ParamTrans<
    TensorWrapper<%s, %d, MemLayout::COL_MAJOR>>;
template struct ParamTrans<
    TensorWrapper<%s, %d, MemLayout::ROW_MAJOR>>;''' % (FLOAT_MAP[dtype].full,
    order, FLOAT_MAP[dtype].full, order)

      temp_content += '\n\n}'
      self.content.append(temp_content)
