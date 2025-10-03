### LIBERO Datasets
To download the [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds) that we used in our fine-tuning experiments, run the command below. This will download the `Spatial`, `Object`, `Goal`, and `Long` datasets in `RLDS` format, i.e., `libero_spatial_no_noops`, `libero_object_no_noops`, `libero_goal_no_noops`, `libero_10_no_noops`. (`"_no_noops"` stands for no no-op actions, i.e., training samples with near-zero actions are filtered out). These datasets require `~10GB` of memory in total. If needed, see details on how to download the original non-RLDS datasets [here](https://github.com/openvla/openvla?tab=readme-ov-file#libero-setup).

```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

The downloaded dataset can be placed in the `/data` folder. The overall directory structure is as follows:

```
·
├── data
·   ├── modified_libero_rlds
    │   ├── libero_10_no_noops
    │   │   └── 1.0.0  (It contains some json files and 32 tfrecord files)
    │   ├── libero_goal_no_noops
    │   │   └── 1.0.0  (It contains some json files and 16 tfrecord files)
    │   ├── libero_object_no_noops
    │   │   └── 1.0.0  (It contains some json files and 32 tfrecord files)
    │   ├── libero_spatial_no_noops
    │   │   └── 1.0.0  (It contains some json files and 16 tfrecord files)
    └── other benchmarks ...
```