# saltax
## Scope
Core functions of `saltax`.
## Structure
- `instructions`: `wfsim` or `fuse` CSV instruction related, including event generators.
- `match`: tools to match salted truth and reconstucted signals.
- `plugins`: `straxen`-like plugins for salting, equivalently from `raw_records` to `peaklets`.
- `contexts.py`: `strax` context for `saltax`. You will want to use `sxenonnt` (`wfsim` based simulation) or `fxenonnt` (`fuse` based simulation) as default context.
