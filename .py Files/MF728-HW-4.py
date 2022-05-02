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
        return f'- F_0  = {self.F}\n- strike = {self.K}\n- expiry = {self.T}\n- volatility = {self.sigma}\n- the annuity = {self.annuity}\n'

    def instentaniousForward(self, F, bps):
        return 2 * np.log(0.5 * F * bps + 1)

    def annuityFunc(self, f, S):
        annuity = np.zeros(len(f))
        for i in range(1, 11):
            annuity += 0.5 * np.exp(-0.5 * (S + i) * f)
        return annuity

    def bachelier(self, F, K, T, sigma, annuity):
        d1 = (F - K) / (sigma * (T ** 0.5))
        return annuity * sigma * (T ** 0.5) * (d1 * si.norm.cdf(d1) + si.norm.pdf(d1))

    def blacksModel(self, F, K, T, sigma, annuity):
        d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(F / K) + (-0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return annuity * si.norm.cdf(d1), annuity * (F * si.norm.cdf(d1) - K * si.norm.cdf(d2))

    def blackVol(self, F0, K, T, sigma, annuity, premium, bps):
        bvol = np.zeros((5, 6))
        for i in range(len(sigma)):
            for j in range(len(sigma[0])):
                bvol[i][j] = root(lambda x: (self.blacksModel(F0[i] * bps, K[i][j] * bps, T, x, annuity[i])[1] - premium[i][j]), 0.01).x
        return bvol

    def objectiveSABR(self, params, T, Klst, F, sigmaLst):
        sigma0, alpha, rho = params
        beta = 0.5
        value = 0
        for i in range(len(Klst)):
            value += (self.asymptoticSABR(F, Klst[i], T, sigma0, alpha, beta, rho) - sigmaLst[i]) ** 2
        return value

    def asymptoticSABR(self, F, K, T, sigma0, alpha, beta, rho):
        Fmid = (F + K) / 2                                                                     # F_(mid) = (F_0 + K) / 2
        gamma1 = beta / Fmid                                                                   # gamma_1
        gamma2 = beta * (beta - 1) / Fmid ** 2                                                 # gamma_2
        zeta = alpha / (sigma0 * (1 - beta)) * (F ** (1 - beta) - K ** (1 - beta))             # zeta from slide 37
        epsilon = T * (alpha ** 2)                                                             # T(alpha^2)
        delta = np.log((np.sqrt(1 - 2 * rho * zeta + zeta ** 2) + zeta - rho) / (1 - rho))     # delta(K, F_0, sigma_0, alpha, beta)

        partA = alpha * (F - K) / delta                                                        # alpha *(F-K)/delta
        part1 = ((2 * gamma2 - gamma1 ** 2) / 24) * ((sigma0 * (Fmid ** beta)) / alpha) ** 2   # fractions 1 & 2 from eq. 27
        part2 = (rho * gamma1 / 4) * ((sigma0 * (Fmid ** beta)) / alpha)                       # fractions 3 & 4 from eq. 27
        part3 = (2 -3 * (rho ** 2)) / 24                                                       # fraction 5 from eq. 27
        asympAprox = (1 + (part1 + part2 + part3) * epsilon)
        return partA * asympAprox

    def blackDeltas(self, F0, K, bsvol, annuity):
        bsdelta = np.zeros((5, 6))
        for i in range(len(K)):
            for j in range(len(K[0])):
                bsdelta[i][j] = self.blacksModel(F0[i] * bps, K[i][j] * bps, 5, bsvol[i][j], annuity[i])[0]
        return bsdelta

    def smileAdjDeltas(self, F0, K, sigma, bps, annuity, param):
        adjdelta = np.zeros((5, 6))
        for i in range(len(sigma)):
            fUp = (F0[i] + 1) * bps
            fDown = (F0[i] - 1) * bps
            for j in range(len(sigma[0])):
                sigmaUp = self.asymptoticSABR(fUp, K[i][j] * bps, 5, param[i][0], param[i][1], 0.5, param[i][2])
                sigmaDown = self.asymptoticSABR(fDown, K[i][j] * bps, 5, param[i][0], param[i][1], 0.5, param[i][2])

                vUp = self.bachelier(fUp, K[i][j] * bps, 5, sigmaUp, annuity[i])
                vDown = self.bachelier(fDown, K[i][j] * bps, 5, sigmaDown, annuity[i])

                vOne = self.bachelier(fUp, K[i][j] * bps, 5, sigma[i][j], annuity[i])
                vTwo = self.bachelier(fDown, K[i][j] * bps, 5, sigma[i][j], annuity[i])
                adjdelta[i][j] = (vOne - vTwo) / (fUp - fDown) + (vUp - vDown) / (fUp - fDown)
        return adjdelta

if __name__ == '__main__':      # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    F0 = np.array([117.45, 120.60, 133.03, 152.05, 171.85])
    S = np.array([1,2,3,4,5])
    bps = 0.0001
    T = 5
    A = 4.813959
    F = 117.45
    K = 120
    sigma = np.array([[57.31, 51.51, 49.28, 48.74, 41.46, 37.33], \
                      [51.72, 46.87, 43.09, 42.63, 38.23, 34.55], \
                      [46.29, 44.48, 43.61, 39.36, 35.95, 32.55], \
                      [45.72, 41.80, 38.92, 38.19, 34.41, 31.15], \
                      [44.92, 40.61, 37.69, 36.94, 33.36, 30.21]])
    # test for the SABR parameters
    sabr = SABR(F, K, T, 57.31, A)
    print(repr(sabr))

    f = sabr.instentaniousForward(F0, bps)
    instentanious = pd.Series(f.T, index= ['1Y','2Y','3Y','4Y','5Y'])
    print(f'Inst. Forward Rates: \n{instentanious}')

    annuity = sabr.annuityFunc(f, S)
    annuity = pd.Series(annuity.T, index= ['1Y','2Y','3Y','4Y','5Y'])
    print(f'\nThe Annuity Factors:  \n{annuity}')

    F = np.array(6 * [117.45, 120.60, 133.03, 152.05, 171.85]).reshape(6, 5).T
    shift = np.array([-50, -25, -5, 5, 25, 50])
    K = F + shift

    #calculate premium
    premium = np.zeros((5, 6))

    for i in range(len(K)):
        for j in range(len(K[0])):
            #  bachelier(F, K, T, sigma, annuity):
            premium[i][j] = sabr.bachelier(F0[i] * bps, K[i][j] * bps, T, sigma[i][j] * bps, annuity[i])
    prem = pd.DataFrame(premium, index=['1Y','2Y','3Y','4Y','5Y'],columns=['ATM-50','ATM-25','ATM-5','ATM+5','ATM+25','ATM+50'])
    print(f'\n---------------------------Premiums---------------------------\n{prem}\n')

    start = [0.1, 0.1, -0.5]
    param = np.zeros((5,3))
    for i in range(len(K)):
        opt = minimize(sabr.objectiveSABR, start, args=(T, K[i] * bps, F0[i] * bps, sigma[i] * bps), method = 'SLSQP', bounds = ((0.01,1.5),(0,1.5),(-1,1)))
        param[i] = opt.x
        print(f'Minimized Vol: \n{opt.fun}')

    SABRparameters = pd.DataFrame(param, index = ['1Y','2Y','3Y','4Y','5Y'], columns = ['sigma0','alpha','rho'])
    print(f'\n------------Parameters----------- \n{SABRparameters}\n')

    F2 = np.array(2 * [117.45, 120.60, 133.03, 152.05, 171.85]).reshape(2, 5).T
    shift2 = np.array([-75, 75])
    K2 = F2 + shift2
    sigma2 = np.zeros((5, 2))
    bachelier2 = np.zeros((5, 2))

    for i in range(len(K2)):
        for j in range(len(K2[0])):
            sigma2[i][j] = sabr.asymptoticSABR(F0[i] * bps, K2[i][j] * bps, T, param[i][0], param[i][1], 0.5, param[i][2])

    K2 = F2 - shift2
    for i in range(len(K2)):
        for j in range(len(K2[0])):
            bachelier2[i][j] = sabr.bachelier(F0[i] * bps, K2[i][j] * bps, T, sigma2[i][j], annuity[i])

    sig2 = pd.DataFrame(sigma2, index=['1Y', '2Y', '3Y', '4Y', '5Y'], columns=['ATM+75', 'ATM-75'])
    price2 = pd.DataFrame(bachelier2, index=['1Y', '2Y', '3Y', '4Y', '5Y'], columns=['ATM+75', 'ATM-75'])
    print(f"\n------Normal Vols------\n{sig2}\n--------Prices--------\n{price2}")

    bsvol = sabr.blackVol(F0,K,T,sigma,annuity,premium,bps)
    bssigma2 = pd.DataFrame(bsvol, index=['1Y', '2Y', '3Y', '4Y', '5Y'], columns=['ATM-50', 'ATM-25', 'ATM-5', 'ATM+5', 'ATM+25', 'ATM+50'])
    print(f"\n---------------------------BS Vols---------------------------\n{bssigma2}")

    bsdelta = sabr.blackDeltas(F0, K, bsvol, annuity)
    bsdel = pd.DataFrame(bsdelta, index=['1Y', '2Y', '3Y', '4Y', '5Y'], columns=['ATM-50', 'ATM-25', 'ATM-5', 'ATM+5', 'ATM+25', 'ATM+50'])
    print(f"\n---------------------------BS Deltas---------------------------\n{bsdel}")

    adjdelta = sabr.smileAdjDeltas(F0, K, sigma, bps, annuity, param)
    smileDeltas = pd.DataFrame(adjdelta, index=['1Y', '2Y', '3Y', '4Y', '5Y'], columns=['ATM-50', 'ATM-25', 'ATM-5', 'ATM+5', 'ATM+25', 'ATM+50'])
    print(f'\n-------------------Smile Adjusted Deltas--------------------\n{smileDeltas}')

