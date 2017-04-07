#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gen_util.gen_types import (FloatType, FLOAT_MAP)


TARGET_PREFIXS = [ ('cgraph_trans', 'CGraphTrans', 'cgraph/cgraph_trans'),
        ('plan_trans', 'PlanTrans', 'plan/plan_trans'),
        ('plan_trans_util', 'PlanTransOptimizer', 'plan/plan_trans') ]

class IncTarget(object):
  def __init__(self, **kwargs):
    dtypes = kwargs['dtype']
    orders = kwargs['order']
    suffix = kwargs['suffix']

    self.filename = []
    self.content = []

    for target in TARGET_PREFIXS:
      self.filename.append('%s_%s' % (target[0], suffix))
      self.content.append('''#pragma once
#ifndef HPTT_GEN_%s_GEN_TCC_
#define HPTT_GEN_%s_GEN_TCC_
''' % (target[0].upper(), target[0].upper()))

      for dtype in dtypes:
        for order in orders:
          self.content[-1] += '''
extern template class %s<ParamTrans<
    TensorWrapper<%s, %d>, true>>;
extern template class %s<ParamTrans<
    TensorWrapper<%s, %d>, false>>;''' % (target[1], FLOAT_MAP[dtype].full,
    order, target[1], FLOAT_MAP[dtype].full, order)

      self.content[-1] += '\n\n#endif'


class SrcTarget(object):
  def __init__(self, **kwargs):
    dtypes = kwargs['dtype']
    orders = kwargs['order']
    suffix = kwargs['suffix']

    self.filename = []
    self.content = []

    for target in TARGET_PREFIXS:
      for dtype in dtypes:
        self.filename.append('%s_%s_%s' % (target[0], FLOAT_MAP[dtype].abbrev,
            suffix))
        self.content.append('''#include <hptt/%s.h>

#include <hptt/types.h>
#include <hptt/tensor.h>
#include <hptt/param/parameter_trans.h>

namespace hptt {
''' % target[2])

        for order in orders:
          self.content[-1] += '''
template class %s<ParamTrans<
    TensorWrapper<%s, %d>, true>>;
template class %s<ParamTrans<
    TensorWrapper<%s, %d>, false>>;''' % (target[1],
    FLOAT_MAP[dtype].full, order, target[1], FLOAT_MAP[dtype].full, order)

        self.content[-1] += '\n\n}'
