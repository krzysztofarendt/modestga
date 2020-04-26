from multiprocessing import Process, Queue
import queue
import os
import logging

import pandas as pd
import numpy as np
import cloudpickle

import modestga
from modestga.ga import OptRes
from modestga.benchmark.functions import rastrigin

logging.basicConfig(level='DEBUG', format="[%(processName)s] %(message)s")


class ParallelMinimize:
    """Runs multiple minimization processes in parallel and returns all results.

    This is the most simple implementation in which there is no communication
    between processes. Instead, the best final solution is returned.
    """
    def __init__(self, fun, bounds, x0=None, args=(),
                 callback=None, options={}, workers=os.cpu_count()):

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
        for p in processes:
            p.join()

        # Read the results
        not_empty = True
        while not_empty:
            try:
                res = self.queue.get(False)
                all_results.append(res)
            except queue.Empty:
                not_empty = False

        # All results to dataframe
        df = pd.DataFrame(all_results).sort_values('fx')

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

        queue.put(result_dict)


def minimize(fun, bounds, x0=None, args=(), callback=None, options={},
             workers=os.cpu_count()):
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
    # Some unpickable (by the standard pickle module) function
    fun = lambda x: np.sum(np.array(x) ** 2)
    pickled = cloudpickle.dumps(fun)

    # Bounds
    bounds = [(-5.12, 5.12) for i in range(10)]

    options = {
        'generations': 100
    }

    def callback(x, fx, ng, *args):
        """Callback function called after each generation"""
        print('CALLBACK')
        print('    Generation #{}'.format(ng))
        print('    x = {}'.format(x))
        print('    fx = {}'.format(fx))

    # pmin = ParallelMinimize(fun, bounds, callback=callback, options=options)
    # res = pmin.run()
    # print(res)

    res = minimize(fun, bounds, callback=callback, options=options)
    print(res)