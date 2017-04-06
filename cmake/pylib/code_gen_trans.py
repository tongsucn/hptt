#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, shutil, sys, argparse

from gen_util import (tensor_gen, core_trans_gen, operation_trans_gen,
    param_trans_gen, hptt_trans_gen)
from gen_util.gen_types import (FloatType, FLOAT_MAP)


class GenTarget(object):
  def __init__(self, target_dir, dtype_usage, order_min, order_max,
        target_suffix = 'gen.cc'):
    if 0 == len(target_dir) or 'gen_util' == target_dir:
      target = 'gen'

    self.target_dir = os.path.join(os.getcwd(), target_dir)
    self.dtype_usage = list(dtype_usage)

    # Check limit values
    if order_min < 2:
      order_min = 2
    if order_max <= order_min:
      order_max = order_min + 1
    self.order_range = list(range(order_min, order_max))
    self.target_suffix = target_suffix

    # Check output directory, previously generated code will be remove
    if os.path.isdir(self.target_dir):
      shutil.rmtree(self.target_dir)
    os.makedirs(self.target_dir)

    self.inc_targets = [ tensor_gen.IncTarget,
        core_trans_gen.IncTarget,
        operation_trans_gen.IncTarget,
        param_trans_gen.IncTarget,
        hptt_trans_gen.IncTarget ]

    self.src_targets = [ tensor_gen.SrcTarget,
        core_trans_gen.SrcTarget,
        operation_trans_gen.SrcTarget,
        param_trans_gen.SrcTarget ]

  def gen(self):
    self.gen_(self.inc_targets, 'gen.tcc')
    self.gen_(self.src_targets, 'gen.cc')

  def gen_(self, targets, suffix):
    print('Creating targets: %d' % len(targets))
    print('Target dir: %s' % self.target_dir)
    for Target in targets:
      tar = Target(dtype = self.dtype_usage, order = self.order_range,
          suffix = suffix)
      for file_idx in range(len(tar.filename)):
        out_file = os.path.join(self.target_dir, tar.filename[file_idx])
        print(out_file)
        target_file = open(out_file, 'w')
        target_file.write(tar.content[file_idx])
        target_file.close()


def arg_parser(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--target', action='store', dest='target_dir')
  parser.add_argument('--order-min', action='store', dest='order_min')
  parser.add_argument('--order-max', action='store', dest='order_max')

  parsed = parser.parse_args(argv[1:])
  parsed.order_min = int(parsed.order_min)
  parsed.order_max = int(parsed.order_max) + 1

  return parsed


def main():
  parsed = arg_parser(sys.argv)
  dtype = [ FloatType.FLOAT, FloatType.DOUBLE, FloatType.FLOAT_COMPLEX,
          FloatType.DOUBLE_COMPLEX ]

  gen_target = GenTarget(parsed.target_dir, dtype, parsed.order_min,
    parsed.order_max)
  gen_target.gen()


if __name__ == '__main__':
  main()
