import numpy as np 
import pandas as pd 

file = open('data/poland_warszawa_2019_ursynow.pb')
all_lines = file.readlines()
print(all_lines[81:,:])