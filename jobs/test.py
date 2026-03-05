from utilix import batchq

import fuse

batchq.submit_job(
    jobstring="echo hello",
    log=f"./test.log",
    partition="lgrandi",
    qos="lgrandi",
    account="pi-lgrandi",
    jobname="test",
    dry_run=True,
    mem_per_cpu=1000,
    container="xenonnt-el8.2026.02.2.simg",
    # bind=None,
    cpus_per_task=1,
    bypass_validation=[""],
)
