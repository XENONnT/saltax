# Patches

This module is to modify the source code of strax(en) to be compatible with the channels augmentation technique. Every modification depends on strax(en) version should be included here.

## Over write strax(en) functions / classes

Replacing the strax(en) functions will not make anything wrong because the simulated channel list have zero overlapping with real channel list.

## How to patch functions / classes

1. Make sure `patches` module is imported first in `saltax`.
2. Get source codes by `inspect`.
3. Modify the codes.
4. Make sure the functions / classes are patched to all possible modules it is defined or reimported.

## Compatible version

| Version | `straxen_2` | `straxen_3` |
| :---: | :---: | :---: |
| `base_environment` | el7.sr1_wimp_unblind | el7.2025.05.2 |
| `strax` | v1.6.5 | v2.2.0 |
| `straxen` | v2.2.7 | v3.2.1 |
| `cutax` | v1.19.5 | v2.2.0 |

`saltax` is also compatible with these branches, with `SingleThreadProcessor` support.

| Version | `straxen_2` |
| :---: | :---: |
| `strax` | v1.6.5-hybrid |
| `straxen` | v2.2.4-hybrid |
| `cutax` | v1.19.1-hybrid |
