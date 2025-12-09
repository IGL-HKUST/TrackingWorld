import os
import signal
import socket
import sys


def sig_handler(signum, frame):
    print("caught signal", signum)
    print(socket.gethostname(), "USR1 signal caught.")
    # do other stuff to cleanup here
    print("requeuing job " + os.environ["SLURM_JOB_ID"])
    os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
    sys.exit(-1)


def term_handler(signum, frame):
    print("bypassing sigterm", flush=True)
