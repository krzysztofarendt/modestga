#!/bin/bash
source venv/bin/activate

python ./modestga/examples/min_noisy_fun.py
python ./modestga/examples/min_rastrigin_fun.py
python ./modestga/examples/min_smooth_fun.py
python ./modestga/examples/con_min_mishra_fun.py
python ./modestga/examples/con_min_rastrigin_fun.py
python ./modestga/examples/con_min_rosenbrock_fun.py
