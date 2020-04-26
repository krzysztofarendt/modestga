from multiprocessing import Process, Queue
import queue
import os
import logging
import time

import pandas as pd
import numpy as np
import cloudpickle

import modestga
from modestga.ga import OptRes
from modestga.benchmark.functions import rastrigin

# logging.basicConfig(level='INFO', format="[%(processName)s] %(message)s")


class ParallelMinimize:
    """Runs multiple minimization processes in parallel and returns all results.

    This is the most simple implementation in which there is no communication
    between processes. Instead, the best final solution is returned.
    """
    def __init__(self, fun, bounds, x0=None, args=(),
                 callback=None, options={}, workers=os.cpu_count() - 1):

        self.pickled_fun = cloudpickle.dumps(fun)
        self.bounds = bounds
        self.x0 = x0
        self.pickled_args = cloudpickle.dumps(args)
        self.pickled_callback = cloudpickle.dumps(callback)
        self.options = options
        self.workers = workers
        self.queue = Queue()

    def run(self):
        processes = list()
        all_results = list()
        for i in range(self.workers):
            pid = i
            p = Process(target=self._optimization_process,
                        args=(self.pickled_fun,
                              self.bounds,
                              self.x0,
                              self.pickled_args,
                              self.pickled_callback,
                              self.options,
                              self.queue,
                              pid)
            )
            p.start()
            processes.append(p)

        # Wait for the results
        n_results = 0

        while n_results < self.workers:
            try:
                res = self.queue.get(True, timeout=5)
                all_results.append(res)
                n_results = len(all_results)
            except queue.Empty:
                time.sleep(0.05)
                continue

        # Join processes
        for p in processes:
            p.join()

        # All results to dataframe
        df = pd.DataFrame(all_results).sort_values('fx')
        
        logging.info(f"Results from all processes:\n{df}")

        return df

    def _optimization_process(self, pickled_fun, bounds, x0,
                              pickled_args, pickled_callback, options,
                              queue, pid):
        fun = cloudpickle.loads(pickled_fun)
        args = cloudpickle.loads(pickled_args)
        callback = cloudpickle.loads(pickled_callback)

        result = modestga.minimize(fun, bounds, x0, args, callback, options)

        result_dict = {
            'pid': pid,
            'x': result.x,
            'message': result.message,
            'ng': result.ng,
            'nfev': result.nfev,
            'fx': result.fx
        }

        queue.put(result_dict, block=True, timeout=5)


def minimize(fun, bounds, x0=None, args=(), callback=None, options={},
             workers=os.cpu_count() - 1):
    """Wrapper over ParallelMinimize providing same interface as modestga.minimize()"""

    pmin = ParallelMinimize(fun, bounds, x0, args, callback, options, workers)
    res = pmin.run()
    best_res = res.iloc[0]  # Beacuse df is sorted by fx in ascending order

    res = OptRes(
        x = best_res['x'],
        message = best_res['message'],
        ng = best_res['ng'],
        nfev = best_res['nfev'],
        fx = best_res['fx']
    )
    return res


if __name__ == "__main__":
    # Example
    fun = rastrigin
    bounds = [(-5.12, 5.12) for i in range(32)]
    options = {
        'generations': 50,
        'pop_size': 100
    }
    def callback(x, fx, ng, *args):
        """Callback function called after each generation"""
        # print(f"\nCallback example:\nx=\n{x}\nf(x)={fx}\n")
        pass

    res = minimize(fun, bounds, callback=callback, options=options)
    print(res)
