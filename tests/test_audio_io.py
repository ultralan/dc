from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from uca8.utils import audio_io


class AudioIOTests(unittest.TestCase):
    def test_disk_cache_reuses_decoded_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "tone.wav"
            cache_dir = root / "cache"
            sample_rate = 8000
            t = np.linspace(0.0, 0.1, num=int(sample_rate * 0.1), endpoint=False)
            waveform = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
            wavfile.write(source, sample_rate, waveform)

            audio_io._load_audio_file_cached.cache_clear()
            first_waveform, first_rate = audio_io.load_audio_file(
                source,
                target_sample_rate=16000,
                cache_dir=cache_dir,
            )
            self.assertEqual(first_rate, 16000)
            self.assertTrue(any(cache_dir.glob("*.pt")))

            audio_io._load_audio_file_cached.cache_clear()
            with patch(
                "uca8.utils.audio_io._decode_audio_file",
                side_effect=AssertionError("disk cache was not used"),
            ):
                second_waveform, second_rate = audio_io.load_audio_file(
                    source,
                    target_sample_rate=16000,
                    cache_dir=cache_dir,
                )

            self.assertEqual(second_rate, 16000)
            self.assertEqual(tuple(first_waveform.shape), tuple(second_waveform.shape))
            self.assertTrue(np.allclose(first_waveform.numpy(), second_waveform.numpy()))


if __name__ == "__main__":
    unittest.main()
