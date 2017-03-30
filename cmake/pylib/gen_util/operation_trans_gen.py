#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gen_util.gen_types import (FloatType, FLOAT_MAP)


TARGET_PREFIX = 'operation_trans'

class IncTarget(object):
  def __init__(self, **kwargs):
    orders = kwargs['order']
    suffix = kwargs['suffix']

    self.filename = ['%s_%s' % (TARGET_PREFIX, suffix)]
    temp_content = '''#pragma once
#ifndef HPTC_GEN_%s_GEN_TCC_
#define HPTC_GEN_%s_GEN_TCC_
''' % (TARGET_PREFIX.upper(), TARGET_PREFIX.upper())

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
