import cloudpickle
import numpy as np
import logging

from modestga import operators
from modestga import population
from modestga import individual
from modestga.ga import norm


def parallel_pop(pipe,
                 pickled_fun,
                 args,
                 bounds,
                 pop_size,
                 trm_size,
                 xover_ratio,
                 mut_rate,
                 end_event):
    """Subpopulation used in parallel GA."""
    logging.debug("Starting process")

    # Unpickle function
    fun = cloudpickle.loads(pickled_fun)

    # Initialize population
    pop = population.Population(pop_size, bounds, fun, evaluate=False)

    while not end_event.is_set():
        # Get data
        try:
            data = pipe.recv()
        except EOFError:
            break
        scale = data['scale']
        pop.set_genes(data['genes'])
        pop.set_fx(data['fx'])

        # Generate children
        children = list()
        fx = list()

        while len(children) < pop_size:
            #Cross-over
            i1, i2 = operators.tournament(pop, trm_size)
            child = operators.crossover(i1, i2, xover_ratio)

            # Mutation
            child = operators.mutation(child, mut_rate, scale)

            # Evaluate f(x)
            child.evaluate()

            # Add to children
            children.append(child)
            fx.append(child.val)

        # Return data (new genes) to the main process
        pop.ind = children
        data = dict()
        data['genes'] = pop.get_genes()
        data['fx'] = fx
        pipe.send(data)

    pipe.close()