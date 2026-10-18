import unittest

import torch

from ml.device import release_cached_memory


class ReleaseCachedMemoryTests(unittest.TestCase):
    @unittest.skipUnless(torch.backends.mps.is_available(), "needs the Apple GPU")
    def test_freed_gpu_buffers_go_back_to_the_system(self):
        buffer = torch.ones(64, 1024, 1024, device="mps")  # 256 MB
        del buffer
        torch.mps.synchronize()
        cached = torch.mps.driver_allocated_memory()
        # Below the limit the cache is kept, to be used again.
        release_cached_memory(torch.device("mps"), limit=cached + 2**30)
        self.assertEqual(torch.mps.driver_allocated_memory(), cached)
        release_cached_memory(torch.device("mps"), limit=0)
        self.assertLess(torch.mps.driver_allocated_memory(), cached - 200 * 2**20)


if __name__ == "__main__":
    unittest.main()
