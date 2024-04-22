# Jobs
## Scope
Job submission scripts.
## Structure
`config.ini`: Configuraiton that you want to change everytime before submitting.
`submit.py`: wrapper around `utilix.batchq` to submit jobs.
`job.py`: processing script, just a wrapper around `st.make`.
## Usage
Update `config.ini` and run this in a container
```
python submit.py
```
## Tips
- Unless you are working on AmBe, put `mem_per_cpu = 45000` (MB) should be enough. Otherwise please do `55000`.
- In `package` you can put either `wfsim` or `fuse`. They are both supported but of course `fuse` is preferred. Please make sure you have the master branch `wfsim` to avoid any photon timing bug.
- When deciding `generator_name`, take a look in `saltax/saltax/instructions/README.md`.
- In `recoil`, put NESTID. (`8` is `beta`, `7` is `gamma`, `0` is `WIMP`).
- Make sure you put `output_folder` to be your own scratch. We will have very huge output because of `records` and `raw_records_simu`.
- `rate` is in unit of Hz. Be cautious to put anything above 300Hz, since you will start to have time gap between sprinkled event less than the Full Drift Time.
- `simu_mode = `all` means you want to simulate both S1 and S2. Otherwise you specify `s1` or `s2`
- Do NOT put space when specifying `runids`. Also don't put `0` in front of numbers!
- I never tested `cpus_per_task != 1`. Please be cautious if you specify another number.
- Make sure you change the `username`, `storage_to_patch` and `output_folder` etc before using.
- In `saltax_mode`, you can put `salt`, `simu` and `data`. Remember that all of them will be computed from the same `records`, so usually you don't want to change `saltax_mode = salt`.
- When running on `dali`, make sure you put `log_dir` in somewhere under `/dali`.