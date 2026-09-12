import unittest
from hardware import parse_cpuinfo, parse_memory, parse_gpus

class HardwareTests(unittest.TestCase):
    def test_isa_and_topology(self):
        def cpu(flags):
            return '\n\n'.join(f'processor : {i}\nphysical id : 0\ncore id : {i%2}\nflags : {flags}' for i in range(4))
        self.assertEqual(parse_cpuinfo(cpu('sse2'))['isa_from_flags'], 'scalar')
        avx2='avx2 fma f16c'
        report=parse_cpuinfo(cpu(avx2))
        self.assertEqual(report['isa_from_flags'], 'avx2')
        self.assertEqual(report['suggested_threads'], 2)
        self.assertEqual(report['logical_available'], 4)
        self.assertEqual(parse_cpuinfo(cpu(avx2+' avx512f avx512dq avx512bw avx512vl'))['isa_from_flags'], 'avx512')
        self.assertEqual(parse_cpuinfo(cpu('avx2 f16c'))['isa_from_flags'], 'scalar')
        self.assertEqual(parse_cpuinfo(cpu(avx2), {0})['suggested_threads'], 1)

    def test_memory_and_gpu_units(self):
        report=parse_memory('MemTotal: 1024 kB\nMemAvailable: 512 kB\nSwapTotal: 0 kB')
        self.assertEqual(report['wsl_total_bytes'], 1048576)
        self.assertEqual(report['wsl_available_bytes'], 524288)
        gpu=parse_gpus('0, GPU-abc, NVIDIA example, 12282, 8192, 8.9, 610.88')[0]
        self.assertEqual(gpu['cuda_arch'], 'sm_89')
        self.assertEqual(gpu['total_bytes'], 12282*1024**2)
        with self.assertRaises(ValueError):parse_gpus('missing fields')

if __name__=='__main__':unittest.main()
