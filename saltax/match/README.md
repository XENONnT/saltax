# Match

## Scope

Functionality for matching salt and simu, with some regular follow-up analysis functions.

## Structure

- `match.py`: core functions for matching simulation and sprinkled dataset. The core functions to use are `match_peaks` or `match_events`.
- `utils.py`: useful functions for analysis after matching is finished. Check tutorial notebooks for usage.
- `visual.py`: a toy non-interactive event viewer. The core functinos to use are `plot_event_wf_w_data` or `plot_event_wf_wo_data`. (The only discrepancy is whether you want to show data-only reconstruction result)
