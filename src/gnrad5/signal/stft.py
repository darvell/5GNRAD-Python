from __future__ import annotations

import numpy as np

from gnrad5.signal.windows import get_dft_window


def stft(
    data,
    window: str = "rect",
    window_len: int = 64,
    overlap: float = 0.5,
    nfft: int = 128,
    dim: int = 2,
):
    if window_len > nfft:
        raise ValueError("FFT size must be greater than or equal to the window length")
    if not (0 <= overlap <= 1):
        raise ValueError("overlap must be in [0, 1]")

    data = np.asarray(data)
    if dim == 1:
        data = data.T
    if data.ndim == 1:
        data = data[None, :]
    n = data.shape[0]
    in_len = data.shape[1]
    if window_len > in_len:
        raise ValueError("The length of the segments cannot be greater than the length of the input signal")

    win = get_dft_window(window, window_len)
    noverlap = int(np.floor(window_len * overlap))
    hop_length = window_len - noverlap
    n_blocks = int(np.floor((in_len - window_len) / hop_length) + 1)
    idx = np.arange(window_len)[:, None] + hop_length * np.arange(n_blocks)[None, :]
    data_blocks = data[:, idx.ravel()]
    data_blocks = data_blocks.reshape(n, window_len, n_blocks)
    data_blocks *= win[None, :, None]
    data_blocks_fft = np.fft.fftshift(np.fft.fft(data_blocks, nfft, axis=1), axes=1)
    out_raw = data_blocks_fft
    out_mean = np.mean(data_blocks_fft, axis=2)

    if dim == 1:
        out_raw = np.transpose(out_raw, (1, 0, 2))
        out_mean = out_mean.T
    return out_mean, out_raw
