#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gen_util.gen_types import (FloatType, FLOAT_MAP)


TARGET_PREFIX = 'tensor'

class IncTarget(object):
  def __init__(self, **kwargs):
    dtypes = kwargs['dtype']
    orders = kwargs['order']
    suffix = kwargs['suffix']

    self.filename = ['%s_%s' % (TARGET_PREFIX, suffix)]
    temp_content = '''#pragma once
#ifndef HPTC_GEN_%s_GEN_TCC_
#define HPTC_GEN_%s_GEN_TCC_
''' % (TARGET_PREFIX.upper(), TARGET_PREFIX.upper())
    for dtype in dtypes:
      for order in orders:
        temp_content += '''
extern template class TensorWrapper<%s, %d, MemLayout::COL_MAJOR>;''' % (
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
      self.content.append('''#include <hptc/tensor.h>

#include <hptc/types.h>
#include <hptc/arch/compat.h>

namespace hptc {
''')
      for order in orders:
        self.content[-1] += '''
template class TensorWrapper<%s, %d, MemLayout::COL_MAJOR>;''' % (
    FLOAT_MAP[dtype].full, order)

      self.content[-1] += '\n\n}'
