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


flag_map = [ ('avx2','HPTC_ARCH_AVX2'), ('avx', 'HPTC_ARCH_AVX') ]

def main():
  selected_arch = ''
  supported_flags = get_cpu_flags()
  for flag in flag_map:
    if flag[0] in supported_flags:
      selected_arch = flag[1]
      break
  sys.stdout.write(selected_arch)


if __name__ == '__main__':
  main()
