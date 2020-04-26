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

logging.basicConfig(level='DEBUG', format="[%(processName)s][%(levelname)s] %(message)s")


class StandardParallel:
    """Runs multiple minimization processes in parallel and returns all results.

    This is the standard implementation in which processes share their fittest
    individuals among one another. The data to share is extracted
    from within `modestga.minimize()` using a callback function `_proc_callback`.
    Due to this approach the code of `modestga.minimize()` did not need to change
    much to enable multiprocessing.
    """
    param_queue = Queue()
    result_queue = Queue()

    def __init__(self, fun, bounds, x0=None, args=(),
                 callback=None, options={}, workers=os.cpu_count() - 1):

        self.pickled_fun = cloudpickle.dumps(fun)
        self.bounds = bounds
        self.x0 = x0
        self.pickled_args = cloudpickle.dumps(args)
        self.pickled_callback = cloudpickle.dumps(callback)
        self.options = options
        self.workers = workers

        self.n_shared = self.workers  # Number of individuals to share among processes

    def run(self):
        processes = list()
        all_results = list()
        for i in range(self.workers):
            pid = i
            p = Process(target=StandardParallel._optimization_process,
                        args=(self.pickled_fun,
                              self.bounds,
                              self.x0,
                              self.pickled_args,
                              self.pickled_callback,
                              self.options,
                              StandardParallel.param_queue,
                              StandardParallel.result_queue,
                              pid,
                              self.n_shared)
            )
            p.start()
            processes.append(p)

        # Wait for the results
        n_results = 0

        while n_results < self.workers:
            # Discard old parameters
            while (StandardParallel.param_queue.qsize() > max(self.n_shared, self.workers) * 2):
                try:
                    _ = StandardParallel.param_queue.get(True, timeout=0.001)
                except queue.Empty:
                    break

            # Try reading all results
            try:
                res = StandardParallel.result_queue.get(True, timeout=0.001)
                all_results.append(res)
                n_results = len(all_results)
            except queue.Empty:
                time.sleep(0.001)
                pass

        # Discard all parameters from the param queue
        self.empty_queue(StandardParallel.param_queue)

        # Join processes
        for p in processes:
            p.join()

        # All results to dataframe
        df = pd.DataFrame(all_results).sort_values('fx')

        logging.info(f"Results from all processes:\n{df}")

        return df

    def empty_queue(self, q, timeout=0.1):
        not_empty = True
        while not_empty is True:
            try:
                _ = q.get(True, timeout=timeout)
            except queue.Empty:
                not_empty = False

    @staticmethod
    def _optimization_process(pickled_fun, bounds, x0,
                              pickled_args, pickled_callback, options,
                              param_queue, result_queue, pid, n_shared):
        fun = cloudpickle.loads(pickled_fun)
        args = cloudpickle.loads(pickled_args)
        callback = cloudpickle.loads(pickled_callback)

        result = modestga.minimize(fun, bounds, x0, args, callback, options,
                                   StandardParallel._proc_callback,
                                   param_queue, n_shared)

        result_dict = {
            'pid': pid,
            'x': result.x,
            'message': result.message,
            'ng': result.ng,
            'nfev': result.nfev,
            'fx': result.fx
        }

        result_queue.put(result_dict, block=True, timeout=5)

    @staticmethod
    def _proc_callback(x, fx, ng, fcid, q, n_shared):
        """Inter-process communication callback.

        Each process should pass here their best individual
        and get from it best foreign individuals.

        :param x: best parameters
        :param fx: best function value
        :param ng: current generation
        :param fcid: function call id
        :param q: queue to share data
        :param n_shared: number of foreign individuals to get from queue
        """
        timeout = 0.1
        foreign_data = list()

        # Get data from queue
        while len(foreign_data) < n_shared:
            try:
                foreign_id, foreign_x, foreign_fx = q.get(block=True,
                                                          timeout=timeout)
                if foreign_id != fcid:
                    foreign_data.append((foreign_id, foreign_x, foreign_fx))

                q.put((foreign_id, foreign_x, foreign_fx),
                      block=True, timeout=timeout)
            except queue.Empty:
                break

        # Put current parameters
        q.put((fcid, x, fx), block=True, timeout=timeout)

        return foreign_data


def minimize(fun, bounds, x0=None, args=(), callback=None, options={},
             workers=os.cpu_count() - 1):
    """Wrapper over StandardParallel providing same interface as modestga.minimize()"""

    pmin = StandardParallel(fun, bounds, x0, args, callback, options, workers)
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
    from modestga.benchmark.functions import rastrigin
    fun = rastrigin
    bounds = [(-5.12, 5.12) for i in range(64)]
    options = {
        'generations': 100,
        'pop_size': 100,
        'tol': 1e-3
    }
    def callback(x, fx, ng, *args):
        """Callback function called after each generation"""
        # print(f"\nCallback example:\nx=\n{x}\nf(x)={fx}\n")
        pass

    res = minimize(fun, bounds, callback=callback, options=options)
    print(res)
