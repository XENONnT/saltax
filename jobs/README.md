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
