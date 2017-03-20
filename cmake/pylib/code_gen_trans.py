#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, shutil, sys, argparse

from gen_util import (tensor_gen, cgraph_trans_gen, operation_trans_gen,
    param_trans_gen, plan_trans_util_gen)
from gen_util.gen_types import (FloatType, CoefTrans, FLOAT_MAP, COEF_TRANS_MAP)


class GenTarget(object):
  def __init__(self, target_dir, dtype_usage, coef_usage, order_min,
      order_max, target_suffix = 'gen.cc'):
    if 0 == len(target_dir) or 'gen_util' == target_dir:
      target = 'gen'

    self.target_dir = os.path.join(os.getcwd(), target_dir)
    self.dtype_usage = list(dtype_usage)
    self.coef_usage = list(coef_usage)
    self.order_range = list(range(order_min, order_max))
    self.target_suffix = target_suffix

    # Check output directory, previously generated code will be remove
    if os.path.isdir(self.target_dir):
      shutil.rmtree(self.target_dir)
    os.makedirs(self.target_dir)

    self.inc_targets = [
        tensor_gen.IncTarget,
        cgraph_trans_gen.IncTarget,
        operation_trans_gen.IncTarget,
        param_trans_gen.IncTarget,
        plan_trans_util_gen.IncTarget
        ]

    self.src_targets = [
        tensor_gen.SrcTarget,
        cgraph_trans_gen.SrcTarget,
        operation_trans_gen.SrcTarget,
        param_trans_gen.SrcTarget,
        plan_trans_util_gen.SrcTarget
        ]

  def gen(self):
    self.gen_(self.inc_targets)
    self.gen_(self.src_targets)

  def gen_(self, targets):
    print('Creating targets: %d' % len(targets))
    print('Target dir: %s' % self.target_dir)
    for Target in targets:
      tar = Target(dtype = self.dtype_usage, coef = self.coef_usage,
          order = self.order_range, suffix = self.target_suffix)
      for file_idx in range(len(tar.filename)):
        out_file = os.path.join(self.target_dir, tar.filename[file_idx])
        print(out_file)
        target_file = open(out_file, 'w')
        target_file.write(tar.content[file_idx])
        target_file.close()


def arg_parser(argv):
  dtype_dict = { 's' : FloatType.FLOAT, 'd' : FloatType.DOUBLE,
      'c' : FloatType.FLOAT_COMPLEX, 'z' : FloatType.DOUBLE_COMPLEX }
  coef_dict = { 'alpha' : CoefTrans.USE_ALPHA, 'beta' : CoefTrans.USE_BETA,
      'both' : CoefTrans.USE_BOTH, 'none' : CoefTrans.USE_NONE }

  parser = argparse.ArgumentParser()
  parser.add_argument('--target', action='store', dest='target_dir')
  parser.add_argument('--dtype', action='store', dest='dtype')
  parser.add_argument('--coef', action='store', dest='coef')
  parser.add_argument('--order-min', action='store', dest='order_min')
  parser.add_argument('--order-max', action='store', dest='order_max')

  parsed = parser.parse_args(argv[1:])
  parsed.dtype = map(lambda x: dtype_dict[x], parsed.dtype.split(','))
  parsed.coef = map(lambda x: coef_dict[x], parsed.coef.split(','))
  parsed.order_min = int(parsed.order_min)
  parsed.order_max = int(parsed.order_max)

  return parsed


def main():
  parsed = arg_parser(sys.argv)

  gen_target = GenTarget(parsed.target_dir, parsed.dtype, parsed.coef,
      parsed.order_min, parsed.order_max)
  gen_target.gen()


if __name__ == '__main__':
  main()
