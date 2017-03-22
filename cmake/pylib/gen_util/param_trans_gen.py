#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gen_util.gen_types import (FloatType, CoefTrans, FLOAT_MAP, COEF_TRANS_MAP)


TARGET_PREFIX = 'parameter_trans'

class IncTarget(object):
  def __init__(self, **kwargs):
    dtypes = kwargs['dtype']
    coefs = kwargs['coef']
    orders = kwargs['order']

    self.filename = ['%s_gen.tcc' % TARGET_PREFIX]
    temp_content = '''#pragma once
#ifndef %s
#define %s
''' % (TARGET_PREFIX.upper() + '_GEN_TCC', TARGET_PREFIX.upper() + '_GEN_TCC')
    for dtype in dtypes:
      for coef in coefs:
        for order in orders:
          temp_content += '''
extern template class ParamTrans<
    TensorWrapper<%s, %d, MemLayout::COL_MAJOR>,
    CoefUsageTrans::%s>;
extern template class ParamTrans<
    TensorWrapper<%s, %d, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::%s>;''' % (FLOAT_MAP[dtype].full, order,
    COEF_TRANS_MAP[coef].full, FLOAT_MAP[dtype].full, order,
    COEF_TRANS_MAP[coef].full)

    temp_content += '\n\n#endif'
    self.content = [temp_content]

class SrcTarget(object):
  def __init__(self, **kwargs):
    dtypes = kwargs['dtype']
    coefs = kwargs['coef']
    orders = kwargs['order']
    suffix = kwargs['suffix']

    self.filename = []
    self.content = []

    for dtype in dtypes:
      for coef in coefs:
        self.filename.append('%s_%s_%s_%s' % (TARGET_PREFIX,
            FLOAT_MAP[dtype].abbrev, COEF_TRANS_MAP[coef].abbrev, suffix))
        temp_content = '''#include <hptc/param/parameter_trans.h>

#include <hptc/types.h>
#include <hptc/compat.h>
#include <hptc/util/util_trans.h>

namespace hptc {
'''
        for order in orders:
          temp_content +='''
template struct ParamTrans<
    TensorWrapper<%s, %d, MemLayout::COL_MAJOR>,
    CoefUsageTrans::%s>;
template struct ParamTrans<
    TensorWrapper<%s, %d, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::%s>;
''' % (FLOAT_MAP[dtype].full, order, COEF_TRANS_MAP[coef].full,
    FLOAT_MAP[dtype].full, order, COEF_TRANS_MAP[coef].full)

        temp_content += '\n\n}'
        self.content.append(temp_content)
