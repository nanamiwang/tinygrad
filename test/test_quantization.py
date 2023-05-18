import unittest
import numpy as np
from tinygrad.helpers import getenv
from tinygrad.lazy import Device
from tinygrad.tensor import Tensor, dtypes

class TestQuantization(unittest.TestCase):
  def test_8_bit_quantization(self):
    QK = 32
    f16 = Tensor(np.arange(-QK, QK, QK/32) / 16.0, dtype=dtypes.float32)
    amax = f16.abs().max() / 127.0
    id = (1.0 / amax) if amax.numpy() != 0 else 0.0
    q8_0 = Tensor((f16 * id + 128.0).numpy(), dtype=dtypes.uint8)

    np_amax = np.abs(f16.numpy()).max() / 127.0
    np_id = (1.0 / np_amax) if np_amax != 0 else 0.0
    np_result = (f16.numpy() * np_id  + 128.0).astype(np.uint8)

    np.testing.assert_allclose(q8_0.numpy(), np_result)

  def test_8_bit_dequantization(self):
    QK = 32
    f16 = Tensor(np.arange(-QK, QK, QK/32) / 16.0, dtype=dtypes.float32)
    amax = f16.abs().max() / 127.0
    id = (1.0 / amax) if amax.numpy() != 0 else 0.0
    q8_0 = Tensor((f16 * id + 128.0).numpy(), dtype=dtypes.uint8)
    dq8_0 = (Tensor(q8_0.numpy(), dtype=dtypes.float32) - 128.0) * amax

    np_amax = np.abs(f16.numpy()).max() / 127.0
    np_result = (q8_0.numpy().astype(np.float32) - 128.0) * np_amax

    np.testing.assert_allclose(dq8_0.numpy(), np_result)
    np.testing.assert_allclose(dq8_0.numpy(), f16.numpy(), rtol=0, atol=0.02)

if __name__ == '__main__':
  unittest.main()
