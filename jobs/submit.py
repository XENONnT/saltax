import os
import time
import configparser

import utilix

config = configparser.ConfigParser()
config.read("config.ini")

MAX_NUM_SUBMIT = config.getint("utilix", "max_num_submit")
T_SLEEP = config.getfloat("utilix", "t_sleep")
USER = config.get("slurm", "username")
ACCOUNT = config.get("slurm", "account")
LOG_DIR = config.get("slurm", "log_dir")
CONTAINER = config.get("job", "container")
RUNIDS = [int(runid) for runid in config.get("job", "runids").split(",")]
JOB_TITLE = config.get("slurm", "job_title")
PARTITION = config.get("slurm", "partition")
QOS = config.get("slurm", "qos")
MEM_PER_CPU = config.getint("slurm", "mem_per_cpu")
CPUS_PER_TASK = config.getint("slurm", "cpus_per_task")

os.makedirs(LOG_DIR, exist_ok=True)


class Submit(object):
    def name(self):
        return self.__class__.__name__

    def execute(self, *args, **kwargs):
        eval("self.{name}(*args, **kwargs)".format(name=self.name().lower()))

    def submit(self, loop_over=[], max_num_submit=10, nmax=10000):
        """Submit jobs to slurm."""
        _start = 0
        self.max_num_submit = max_num_submit
        self.loop_over = loop_over
        self.p = True

        index = _start
        while index < len(self.loop_over) and index < nmax:
            if self.working_job() < self.max_num_submit:
                self._submit_single(loop_index=index, loop_item=self.loop_over[index])

                time.sleep(T_SLEEP)
                index += 1

    # check my jobs
    def working_job(self):
        """Check how many jobs are running."""
        cmd = "squeue --user={user} | wc -l".format(user=USER)
        jobNum = int(os.popen(cmd).read())
        return jobNum - 1

    def _submit_single(self, loop_index, loop_item):
        """Submit a single job."""
        jobname = JOB_TITLE + "_{:03}".format(loop_item)
        # Modify here for the script to run
        jobstring = f"python job.py {loop_item}"
        print(jobstring)

        # Modify here for the log name
        utilix.batchq.submit_job(
            jobstring=jobstring,
            log=f"{LOG_DIR}/{jobname}.log",
            partition=PARTITION,
            qos=QOS,
            account=ACCOUNT,
            jobname=jobname,
            dry_run=False,
            mem_per_cpu=MEM_PER_CPU,
            container=CONTAINER,
            cpus_per_task=CPUS_PER_TASK,
        )


p = Submit()

# Modify here for the runs to process
loop_over = RUNIDS
print("Going to process these runs:", loop_over)
print("Number of runs to process: ", len(loop_over))
print("Your log files are in: ", LOG_DIR)
p.execute(loop_over=loop_over, max_num_submit=MAX_NUM_SUBMIT)
