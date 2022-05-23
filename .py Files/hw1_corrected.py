"""
Created on Mon Feb  7 16:36:01 2022

@author: Thomas Kuntz MF728-HW-1
"""

import pandas as pd
import numpy as np
from math import *
import QuantLib as ql
import scipy as sp
import scipy.stats as si
import statsmodels.api as sm
import seaborn as sns
import sympy as sy
from scipy.optimize import newton
from tabulate import tabulate
from pandas_datareader import data
import matplotlib.pyplot as plt
from datetime import datetime
import calendar


# part a
def creditDefaultSwaps(S1, P0, P1, R, r):
    premium = S1/2 * (np.exp(-r * 1) * (P0 + P1))
    contingent = (1 - R) * (np.exp(-r * 1) * (P0 - P1))
    return premium - contingent

















if __name__ == '__main__':      # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # part a














