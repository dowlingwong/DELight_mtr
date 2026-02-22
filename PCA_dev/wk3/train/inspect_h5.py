import h5py
import sys

filename = "/ceph/dwong/trigger_samples/PCA_QP/main/NR_traces_energy_500_pair_qp_sum_batch_0000.h5"
try:
    with h5py.File(filename, "r") as f:
        print(f"Keys in {filename}:")
        print(list(f.keys()))
        for key in f.keys():
            print(f"  {key}: {f[key]}")
except Exception as e:
    print(e)
