from pathlib import Path
import numpy as np
import zstandard as zstd
import h5py


def unshuffle_bytes(data: bytes, dtype, shape) -> np.ndarray:
    """
    Undo the byte-shuffle for a single trace.
    """
    itemsize = np.dtype(dtype).itemsize
    num_elements = int(np.prod(shape))
    reshaped = np.frombuffer(data, dtype=np.uint8).reshape(itemsize, num_elements).T
    unshuffled = reshaped.reshape(-1)
    return unshuffled.view(dtype).reshape(shape)


def zstd_to_h5_chunks(
    input_path: Path,
    output_dir: Path,
    n_traces_total: int,
    trace_shape: tuple[int, int] = (56, 65536),
    traces_per_file: int = 1000,
    dtype=np.float16,
) -> None:
    """
    Stream-decompress a .zst file containing byte-shuffled traces and
    write them into multiple HDF5 files with shape
    (traces_per_file, *trace_shape).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    itemsize = np.dtype(dtype).itemsize
    trace_size_bytes = int(np.prod(trace_shape)) * itemsize

    dctx = zstd.ZstdDecompressor()

    with open(input_path, "rb") as fh, dctx.stream_reader(fh) as reader:
        buffer = b""
        trace_idx_global = 0
        file_idx = 0

        batch_size = traces_per_file
        batch = np.empty((batch_size, *trace_shape), dtype=np.float32)
        batch_idx = 0

        while trace_idx_global < n_traces_total:
            # Read some decompressed bytes
            chunk = reader.read(1024 * 1024)  # 1 MiB
            if not chunk:
                break
            buffer += chunk

            # While we have at least one full trace in the buffer
            while len(buffer) >= trace_size_bytes and trace_idx_global < n_traces_total:
                trace_bytes = buffer[:trace_size_bytes]
                buffer = buffer[trace_size_bytes:]

                # Reconstruct one trace
                trace = unshuffle_bytes(trace_bytes, dtype=dtype, shape=trace_shape).astype(np.float32)
                batch[batch_idx] = trace
                batch_idx += 1
                trace_idx_global += 1

                # If batch is full or we've reached the last trace -> write an HDF5 file
                if batch_idx == batch_size or trace_idx_global == n_traces_total:
                    data_to_write = batch[:batch_idx]  # shrink last batch if needed
                    out_path = output_dir / f"traces_{file_idx:03d}.h5"
                    with h5py.File(out_path, "w") as h5:
                        h5.create_dataset(
                            "traces",
                            data=data_to_write,
                            compression="gzip",
                            compression_opts=4,
                        )
                    print(f"Wrote {data_to_write.shape[0]} traces to {out_path}")
                    file_idx += 1
                    batch_idx = 0  # reset for next batch

        if trace_idx_global != n_traces_total:
            raise RuntimeError(
                f"Expected {n_traces_total} traces, but decoded {trace_idx_global}"
            )


if __name__ == "__main__":
    input_path = Path("/ceph/dwong/work/training_samples/ER/small/traces_energy_500.zst")
    output_dir = Path("/ceph/dwong/work/training_samples/ER/small/split_1k")
    zstd_to_h5_chunks(
        input_path=input_path,
        output_dir=output_dir,
        n_traces_total=50000,
        trace_shape=(56, 65536),
        traces_per_file=1000,
        dtype=np.float16,
    )
