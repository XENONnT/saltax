# Plugins
## Scope
Modified `straxen` [data structure](https://straxen.readthedocs.io/en/latest/reference/datastructure_nT.html) up to `peaklets`. Everything above is designed to be exactly the same as `straxen`.
## Structure
- `s_raw_records.py`: Modified `raw_records` specifically for simulated `raw_records`, which determine chunking time range based on the ones' from `raw_reocrds`
- `s_records.py`: Modified `records`, which combines `raw_reords` and `raw_records_simu` together. The latter one's channel starts at constant variable `SCHANNEL_STARTS_AT`.
- `s_peaklets.py`: Modified `peaklets`, which build peaks as there are `2*n_tpc_pmts` channels, but sum up per-channel information (`area_per_channel` and top-bottom specific fields) as if there are only `n_tpc_pmts`.
## Data Structure
<img width="521" alt="image" src="https://github.com/FaroutYLq/saltax/assets/47046530/9c3ef86c-a171-4082-914a-98f6eee14a58">
Here the numbers are marking number of channels in the plugin.
