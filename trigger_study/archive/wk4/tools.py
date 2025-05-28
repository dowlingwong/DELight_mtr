import numpy as np
import zstandard as zstd
import matplotlib.pyplot as plt

def plot_trace_set(trace_set, offset=50):
    """
    Plot a single trace set (shape: 54 x N) with vertical offsets.

    Parameters:
    - trace_set: ndarray, shape (54, N)
    - offset: float, vertical offset between channels
    """
    n_channels, n_samples = trace_set.shape

    for i in range(n_channels):
        color = 'r' if i > 44 else 'b'
        plt.plot(np.arange(n_samples), trace_set[i] + i * offset, color=color, lw=0.2)

    plt.xlabel("Sample Index")
    plt.yticks([])
    plt.ylim(-10, offset * n_channels)
    plt.title("Trace Set")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def load_traces_from_zstd(input_path, n_traces, dtype=np.float16, trace_shape=(54, 32768)) -> np.ndarray:
    """
    Load a list of numpy arrays (traces) from a compressed Zstandard (.zst) file and return a single stacked ndarray.
    """
    def unshuffle_bytes(data: bytes, dtype=np.float16, shape=(54, 32768)) -> np.ndarray:
        itemsize = np.dtype(dtype).itemsize
        unshuffled = np.frombuffer(data, dtype=np.uint8).reshape(itemsize, -1).T.reshape(-1)
        return unshuffled.view(dtype).reshape(shape)

    decompressor = zstd.ZstdDecompressor()
    with open(input_path, 'rb') as f:
        compressed_content = f.read()
        decompressed = decompressor.decompress(compressed_content)

    trace_size_bytes = np.prod(trace_shape) * np.dtype(dtype).itemsize
    expected_size = n_traces * trace_size_bytes
    if len(decompressed) != expected_size:
        raise ValueError("Decompressed size does not match expected size")

    itemsize = np.dtype(dtype).itemsize
    unshuffled = np.frombuffer(decompressed, dtype=np.uint8)
    unshuffled = unshuffled.reshape(itemsize, -1).T.reshape(-1)
    traces = unshuffled.view(dtype).reshape((n_traces,) + trace_shape)

    return traces