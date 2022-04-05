"""
Created on 02/23/2022

@author: Thomas Kuntz MF728-HW-4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
from scipy.optimize import root, minimize

class SABR:

    def __init__(self, F, K, T, sigma, annuity):
        self.F = F
        self.K = K
        self.T = T
        self.sigma = sigma
        self.annuity = annuity

    def __repr__(self):
        return f'- F_0  = {self.F}\n- stike = {self.K}\n- expiry = {self.T}\n- volatility = {self.sigma}\n- the annuity = {self.annuity}\n'

    def bachelier(self, F, K, T, sigma, annuity):
        d1 = (F - K) / (sigma * (T ** 0.5))
        return annuity * sigma * (T ** 0.5) * (d1 * si.norm.cdf(d1) + si.norm.pdf(d1))

    def blacksModel(self, F, K, T, sigma, annuity):
        d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(F / K) + (-0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return annuity * si.norm.cdf(d1), annuity * (F * si.norm.cdf(d1) - K * si.norm.cdf(d2))

    def objectiveSABR(self, params, T, Klst, F, sigmaLst):
        sigma0, alpha, rho = params
        beta = 0.5
        value = 0
        for i in range(len(Klst)):
            value += (self.asymptoticSABR(F, Klst[i], T, sigma0, alpha, beta, rho) - sigmaLst[i]) ** 2
        return value

    def asymptoticSABR(self, F, K, T, sigma0, alpha, beta, rho):
        midpoint = (F + K) / 2
        zeta = alpha / (sigma0 * (1 - beta)) * (F ** (1 - beta) - K ** (1 - beta))
        epsilon = T * alpha ** 2
        delta = np.log((np.sqrt(1 - 2 * rho * zeta + zeta ** 2) + zeta - rho) / (1 - rho))
        gamma1 = beta / midpoint
        gamma2 = beta * (beta - 1) / midpoint ** 2
        partA = alpha * (F - K) / delta
        partBa = (2 * gamma2 - gamma1 ** 2) / 24 * (sigma0 * midpoint ** beta / alpha) ** 2
        partBb = rho * gamma1 / 4 * sigma0 * midpoint ** beta / alpha
        partBc = (2 -3 * rho ** 2) / 24
        partB = (1 + (partBa + partBb + partBc) * epsilon)
        return partA * partB


if __name__ == '__main__':      # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    F0 = np.array([117.45, 120.60, 133.03, 152.05, 171.85])
    S = np.array([1,2,3,4,5])
    bps = 0.0001

    sabr = SABR(1, 130, 5, 20, 0.76)
    print(repr(sabr))

    f = 2 * np.log(0.5 * F0 * bps + 1)
    instentanious = pd.Series(f.T, index= ['1Y','2Y','3Y','4Y','5Y'], name='Inst. Forwards')
    print(f'Instanteaneous Forward Rates: \n{instentanious}')

    annuity = np.zeros(len(f))
    for i in range(1,11):
        annuity += 0.5 * np.exp(-0.5 * (S + i) * f)
    annuity = pd.Series(annuity.T, index= ['1Y','2Y','3Y','4Y','5Y'], name='Annuities')
    print(f'\nThe annuity factors:  \n{annuity}')

    F = np.array(6 * [117.45, 120.60, 133.03, 152.05, 171.85]).reshape(6, 5).T
    change = np.array([-50, -25, -5, 5, 25, 50])
    K = F + change
    sigma = np.array([[57.31, 51.51, 49.28, 48.74, 41.46, 37.33], \
                      [51.72, 46.87, 43.09, 42.63, 38.23, 34.55], \
                      [46.29, 44.48, 43.61, 39.36, 35.95, 32.55], \
                      [45.72, 41.80, 38.92, 38.19, 34.41, 31.15], \
                      [44.92, 40.61, 37.69, 36.94, 33.36, 30.21]])
    premium = np.zeros((5, 6))

    for i in range(len(K)):
        for j in range(len(K[0])):
            #  bachelier(F, K, T, sigma, annuity):
            premium[i][j] = sabr.bachelier(F0[i] * bps, K[i][j] * bps, 5, sigma[i][j] * bps, annuity[i])
    prem = pd.DataFrame(premium, index=['1Y','2Y','3Y','4Y','5Y'],columns=['ATM-50','ATM-25','ATM-5','ATM+5','ATM+25','ATM+50'])
    print(f'\nPremiums: \n{prem}\n')

    start = [0.1, 0.1, -0.1]
    param = np.zeros((5,3))
    for i in range(len(K)):
        opt = minimize(sabr.objectiveSABR, start, args=(5, K[i] * bps, F0[i] * bps, sigma[i] * bps), method = 'SLSQP', bounds = ((0.01,1.5),(0,1.5),(-1,1)))
        param[i] = opt.x
        print(f'minimized vol: \n{opt.fun}')

    SABRr = pd.DataFrame(param, index = ['1Y','2Y','3Y','4Y','5Y'], columns = ['sigma0','alpha','rho'])
    print(f'\nParameters : \n{SABRr}\n')

    F2 = np.array(2 * [117.45, 120.60, 133.03, 152.05, 171.85]).reshape(2, 5).T
    change2 = np.array([-75, 75])
    K2 = F2 + change2
    sigma2 = np.zeros((5, 2))
    bachelier2 = np.zeros((5, 2))

    for i in range(len(K2)):
        for j in range(len(K2[0])):
            sigma2[i][j] = sabr.asymptoticSABR(F0[i] * bps, K2[i][j] * bps, 5, param[i][0], param[i][1], 0.5, param[i][2])

    K2 = F2 - change2
    for i in range(len(K2)):
        for j in range(len(K2[0])):
            bachelier2[i][j] = sabr.bachelier(F0[i] * bps, K2[i][j] * bps, 5, sigma2[i][j], annuity[i])
    sig2 = pd.DataFrame(sigma2, index=['1Y', '2Y', '3Y', '4Y', '5Y'], columns=['ATM+75', 'ATM-75'])
    price2 = pd.DataFrame(bachelier2, index=['1Y', '2Y', '3Y', '4Y', '5Y'], columns=['ATM+75', 'ATM-75'])
    print(f"Normal Vols are:\n{sig2}\nPrices are:\n{price2}")

    bsvol = np.zeros((5, 6))
    for i in range(len(sigma)):
        for j in range(len(sigma[0])):
            bsvol[i][j] = root(lambda x: (sabr.blacksModel(F0[i] * bps, K[i][j] * bps, 5, x, annuity[i])[1] - premium[i][j]), 0.1).x
    bssigma2 = pd.DataFrame(bsvol, index=['1Y', '2Y', '3Y', '4Y', '5Y'], columns=['ATM-50', 'ATM-25', 'ATM-5', 'ATM+5', 'ATM+25', 'ATM+50'])
    print(f"BS Vols are:\n{bssigma2}")

    bsdelta = np.zeros((5, 6))
    for i in range(len(K)):
        for j in range(len(K[0])):
            bsdelta[i][j] = sabr.blacksModel(F0[i] * bps, K[i][j] * bps, 5, bsvol[i][j], annuity[i])[0]
    bsdel = pd.DataFrame(bsdelta, index=['1Y', '2Y', '3Y', '4Y', '5Y'], columns=['ATM-50', 'ATM-25', 'ATM-5', 'ATM+5', 'ATM+25', 'ATM+50'])
    print(f"BS deltas are:\n{bsdel}")

    adjdelta = np.zeros((5, 6))
    for i in range(len(sigma)):
        f_up = (F0[i] + 1) * bps
        f_down = (F0[i] - 1) * bps
        for j in range(len(sigma[0])):
            sigma_up = sabr.asymptoticSABR(f_up, K[i][j] * bps, 5, param[i][0], param[i][1], 0.5, param[i][2])
            sigma_down = sabr.asymptoticSABR(f_down, K[i][j] * bps, 5, param[i][0], param[i][1], 0.5, param[i][2])
            v_up = sabr.bachelier(f_up, K[i][j] * bps, 5, sigma_up, annuity[i])
            v_down = sabr.bachelier(f_down, K[i][j] * bps, 5, sigma_down, annuity[i])
            v_1 = sabr.bachelier(f_up, K[i][j] * bps, 5, sigma[i][j], annuity[i])
            v_2 = sabr.bachelier(f_down, K[i][j] * bps, 5, sigma[i][j], annuity[i])
            adjdelta[i][j] = (v_1 - v_2) / (f_up - f_down) + (v_up - v_down) / (f_up - f_down)
    smileDeltas = pd.DataFrame(adjdelta, index=['1Y', '2Y', '3Y', '4Y', '5Y'], columns=['ATM-50', 'ATM-25', 'ATM-5', 'ATM+5', 'ATM+25', 'ATM+50'])
    print(f'smile adjusted deltas:  \n{smileDeltas}')

