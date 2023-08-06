# Plugins
## Scope
Modified `straxen` [data structure](https://straxen.readthedocs.io/en/latest/reference/datastructure_nT.html) up to `peaklets`. Everything above is designed to be exactly the same as `straxen`.
## Structure
- `s_raw_records.py`: Modified `raw_records` specifically for simulated `raw_records`, which determine chunking time range based on the ones' from `raw_reocrds`
- `s_records.py`: Modified `records`, which combines `raw_reords` and `raw_records_simu` together. The latter one's channel starts at constant variable `SCHANNEL_STARTS_AT`.
- `s_peaklets.py`: Modified `peaklets`, which build peaks as there are `2*n_tpc_pmts` channels, but sum up per-channel information (`area_per_channel` and top-bottom specific fields) as if there are only `n_tpc_pmts`.
## Data Structure
https://user-images.githubusercontent.com/47046530/253776904-68a96842-5de4-4986-80f8-2555b43114ce.png![image](https://github.com/FaroutYLq/saltax/assets/47046530/5d394752-c406-448b-81ad-1fe2d51c71f1)

