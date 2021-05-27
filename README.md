# The frequency of possibilities for manipulation in participatory budgeting

This repository contains the code with which the results in our paper are obtained. To replicate the results make a directory that is called data/ to store the results in, and if the results are to be replicated based on real-world data make a directory called input_data/ which contains a pb dataset.

To replicate the results for unit costs and theoretical distributions run: `python run_experiments.py --cost_max 2`  <br/>
To replicate the results for varying costs for the theoretical distributions run: `python run_experiments.py --cost_max 11` <br/>
To replicate the results for real-world distributions make run: `python run_experiments.py --path 'path/to/data.pb'`<br/>

Other experiments can be conducted by varying the parameters for `run_experiments.py`, or running specific instances with `strategyproof.py`.
