import numpy as np

from modestga import con_minimize
from modestga.benchmark.functions.mishra_bird import mishra_bird
from modestga.benchmark.functions.mishra_bird import mishra_bird_constr


def test_con_min():
    # 1 worker
    res = con_minimize(
        fun=mishra_bird,
        bounds=[(-10.0, 0.0), (-6.5, 0.0)],
        constr=[mishra_bird_constr],
        workers=1,
        options={"generations": 3},
    )
    # 2 workers
    res = con_minimize(
        fun=mishra_bird,
        bounds=[(-10.0, 0.0), (-6.5, 0.0)],
        constr=[mishra_bird_constr],
        workers=2,
        options={"generations": 3},
    )
