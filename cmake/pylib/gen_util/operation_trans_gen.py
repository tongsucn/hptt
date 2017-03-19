#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gen_util.gen_types import (FloatType, CoefTrans, FLOAT_MAP, COEF_TRANS_MAP)


TARGET_PREFIX = 'operation_trans'

class IncTarget(object):
  def __init__(self, **kwargs):
    orders = kwargs['order']

    self.filename = ['%s_gen.tcc' % TARGET_PREFIX]
    temp_content = '''#pragma once
#ifndef %s
#define %s
''' % (TARGET_PREFIX.upper() + '_GEN_TCC', TARGET_PREFIX.upper() + '_GEN_TCC')

    for order in orders:
      temp_content += '\nextern template class OpForTrans<%d>;' % order
    temp_content += '\n\n#endif'
    self.content = [temp_content]

class SrcTarget(object):
  def __init__(self, **kwargs):
    orders = kwargs['order']
    suffix = kwargs['suffix']

    self.filename = ['%s_%s' % (TARGET_PREFIX, suffix)]
    temp_content = '''#include <hptc/operations/operation_trans.h>

namespace hptc {
'''
    for order in orders:
      temp_content += '\ntemplate class OpForTrans<%d>;' % order
    temp_content += '\n\n}'
    self.content = [temp_content]
