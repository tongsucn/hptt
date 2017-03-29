#!/usr/bin/env python3

import subprocess, sys

def get_cpu_flags():
  pipe = subprocess.Popen(['cat', '/proc/cpuinfo'], stdout=subprocess.PIPE)
  result = pipe.communicate()

  # Check result
  if result[0] is None:
    sys.stderr.write('Cannot get information from /proc/cpuinfo\n')
    sys.stderr.write(result[1])

  result = result[0].decode().split('\n')
  flags = []
  for line in result:
    if 'flags' == line[:5]:
      flags = line.split(':')[-1].split()
      flags = list(map(lambda x: x.lower(), flags))
      break
  return flags


flag_list = [ 'avx2', 'avx']

def main():
  supported_flags = get_cpu_flags()
  candidates = ''
  for flag in flag_list:
    if flag in supported_flags:
      candidates += '%s ' % flag
  sys.stdout.write(candidates[:-1])


if __name__ == '__main__':
  main()
