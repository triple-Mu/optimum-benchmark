import os
import multiprocessing as mp
from logging import getLogger
from typing import Callable, Dict, Any
from multiprocessing import Process, Queue


from ...logging_utils import setup_colorlog_logging
from ..isolation_utils import device_isolation
from .config import ProcessConfig
from ..base import Launcher

LOGGER = getLogger("process")


class ProcessLauncher(Launcher[ProcessConfig]):
    NAME = "process"

    def __init__(self, config: ProcessConfig):
        super().__init__(config)

        if mp.get_start_method(allow_none=True) != self.config.start_method:
            LOGGER.info(f"Setting multiprocessing start method to {self.config.start_method}.")
            mp.set_start_method(self.config.start_method, force=True)

    def launch(self, worker: Callable, *worker_args) -> Dict[str, Any]:
        # worker process can't be daemon since it might spawn its own processes
        queue = Queue()
        worker_process = Process(target=target, args=(worker, queue, *worker_args), daemon=False)
        worker_process.start()
        LOGGER.info(f"\t+ Launched worker process with PID {worker_process.pid}.")

        with device_isolation(enabled=self.config.device_isolation, benchmark_pid=os.getpid()):
            worker_process.join()

        if worker_process.exitcode != 0:
            LOGGER.error(f"Worker process exited with code {worker_process.exitcode}, forwarding...")
            exit(worker_process.exitcode)

        report = queue.get()

        return report


def target(fn, q, *args):
    """This a pickalable function that correctly sets up the logging configuration for the worker process."""

    setup_colorlog_logging()

    out = fn(*args)
    q.put(out)
