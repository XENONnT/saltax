# Plugins

## Scope

Modified `straxen` and `fuse` [data structure](https://straxen.readthedocs.io/en/latest/reference/datastructure_nT.html) up to `peaklets`. Everything above is designed to be exactly the same as `straxen`.

## Structure

- `s_raw_records.py`: `fuse` based plugins. Modified `ChunkCsvInput` from `fuse` specifically for  `raw_records_simu`, which determine chunking time range based on the ones' from `raw_reocrds`
- `records.py`: Modified `records`, which combines `raw_reords` and `raw_records_simu` together. The latter one's channel starts at constant variable `SCHANNEL_STARTS_AT`.
- `peaklets.py`: Modified `peaklets`, which build peaks as there are `2 * straxen.n_tpc_pmts` channels, but sum up per-channel information (`area_per_channel` and top-bottom specific fields) as if there are only `straxen.n_tpc_pmts`.

## Data Structure

<img width="1700" alt="image" src="https://github.com/FaroutYLq/saltax/assets/47046530/5042d5cd-d42f-4a56-8904-b4911c5efe1c">


Here the numbers are marking number of channels in the plugin.
