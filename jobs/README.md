# Jobs

## Scope

Job submission scripts.

## Structure

- `config.ini`: Configuraiton that you want to change everytime before submitting.
- `submit.py`: wrapper around `utilix.batchq` to submit jobs.
- `job.py`: processing script, just a wrapper around `st.make`.

## Usage

Update `config.ini` and run this in a container
```
python submit.py
```

## Logging

When submitting jobs, you might want to have a bit more information for debugging. To do that, please do this before job submission:
- Copy the default xenon config to your home or somewhere safe `/project2/lgrandi/xenonnt/xenon.config`
- Keep everything else the same, and then modify the logging level there by `logging_level=debug` (or `info` should also work). See [here](https://github.com/XENONnT/utilix/blob/b94ef41851e437efa35ae9dc82c6fcdfca77b88c/utilix/config.py#L95) for details.
- Then export the config by `export XENON_CONFIG=<YOUR_DIR_TO_NEW_CONFIG>`.
- Submit jobs as usual.


## Tips

- You need to download yourself `raw_records` and `raw_records_aqmon` before submission!
- Unless you are working on AmBe, put `mem_per_cpu = 45000` (MB) should be enough. Otherwise please do `55000`.
- When deciding `generator_name`, take a look in `saltax/saltax/instructions/README.md`.
- In `recoil`, put NESTID. (`8` is `beta`, `7` is `gamma`, `0` is `WIMP`).
- In en_range, only specify it when you are using a flat spectrum generator. Otherwise keep in mind that it will be treaed as None no matter what you put. Also keep in mind that it is recoil energy, rather than observable energy, so the quenching need to be considered for NR case.
- Make sure you put `output_folder` to be your own scratch. We will have very huge output because of `records` and `raw_records_simu`.
- `rate` is in unit of Hz. Be cautious to put anything above 300Hz, since you will start to have time gap between sprinkled event less than the Full Drift Time.
- `simu_mode = all` means you want to simulate both S1 and S2. Otherwise you specify `s1` or `s2`.
- Do NOT put space when specifying `runids`. Also don't put `0` in front of numbers!
- I never tested `cpus_per_task != 1`. Please be cautious if you specify another number.
- Make sure you change the `username`, `storage_to_patch` and `output_folder` etc before using.
- In `saltax_mode`, you can put `salt`, `simu` and `data`. Remember that all of them will be computed from the same `records`, so usually you don't want to change `saltax_mode = salt`.
- When running on `dali`, make sure you put `log_dir` in somewhere under `/dali`. (Otherwise you will ruin the fragile NFS connection!)
