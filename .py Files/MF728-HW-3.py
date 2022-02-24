"""
Created on 02/23/2022

@author: Thomas Kuntz MF728-HW-3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import *
import scipy as sp
from sympy.stats import Normal, cdf
import scipy.stats as si
from scipy.optimize import root


class base():

    def d1(self,S, K, T, r, sigma):
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def d2(self,S, K, T, r, sigma):
        return (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def euroCall(self, S, K, T, r, sigma):
        call = (S * si.norm.cdf(self.d1(S, K, T, r, sigma), 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(self.d2(S, K, T, r, sigma), 0.0, 1.0))
        return float(call)

    def euroPut(self, S, K, T, r, sigma):
        put = (K * np.exp(-r * T) * si.norm.cdf(-self.d2(S, K, T, r, sigma), 0.0, 1.0) - S * si.norm.cdf(-self.d1(S, K, T, r, sigma), 0.0, 1.0))
        return float(put)

    def discountFactor(self,f,t):
        return 1/(1 + f)**t

class euroGreeks(base):

    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def delta(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        deltaCall = si.norm.cdf(self.d1(self.S, self.K, self.T, self.r, self.sigma), 0.0, 1.0)
        deltaPut = si.norm.cdf(-self.d1(self.S, self.K, self.T, self.r, self.sigma), 0.0, 1.0)
        return deltaCall, -deltaPut

    def gamma(self):
        return (1 / np.sqrt(2 * np.pi) * np.exp(-self.d1(self.S, self.K, self.T, self.r, self.sigma) ** 2 * 0.5)) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * (1 / np.sqrt(2 * np.pi) * np.exp(-self.d1(self.S, self.K, self.T, self.r, self.sigma) ** 2 * 0.5)) * np.sqrt(self.T)

    def theta(self):
        density = 1 / np.sqrt(2 * np.pi) * np.exp(-self.d1(self.S, self.K, self.T, self.r, self.sigma) ** 2 * 0.5)
        cTheta = (-self.sigma * self.S * density) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(self.S, self.K, self.T, self.r, self.sigma), 0.0, 1.0)
        pTheta = (-self.sigma * self.S * density) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(self.S, self.K, self.T, self.r, self.sigma), 0.0, 1.0)
        return cTheta, pTheta

class caplet(base):

    def __init__(self,K,F,sigma,delta,expiry,t):
        self.K = K
        self.F = F
        self.sigma = sigma
        self.delta = delta
        self.expiry = expiry
        self.t = t

    def logNormalCaplet(self):
        d1 = (np.log(self.F / self.K) + (0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = (np.log(self.F / self.K) + (-0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))

        return self.delta * self.discountFactor(self.F, self.expiry) * (self.K * si.norm.cdf(-d2, 0.0, 1.0) - self.F * si.norm.cdf(-d1, 0.0, 1.0))

    def bachelierCaplet(self):
        d = (self.F - self.K) / (self.sigma * np.sqrt(self.t))
        return self.delta * self.sigma * self.discountFactor(self.F,self.expiry) * np.sqrt(self.t) * (-d * si.norm.cdf(-d, 0.0, 1.0) + si.norm.pdf(-d, 0.0, 1.0))

    def volHelper(self,K,F,sigma,delta,expiry,t):
        d = (F - K) / (sigma * np.sqrt(t))
        return delta * sigma * self.discountFactor(F, expiry) * np.sqrt(t) * (-d * si.norm.cdf(-d, 0.0, 1.0) + si.norm.pdf(-d, 0.0, 1.0))

    def impliedVolatility(self,K,F,sigma,delta,expiry,t):
        IV = root(lambda iv: (self.volHelper(K,F,iv,delta,expiry,t) - self.logNormalCaplet()),0.001)
        return IV.x

class capletGreeks(base):

    def __init__(self,K,F,sigma,Delta,expiry,t,lnc,df):
        self.K =K
        self.F = F
        self.sigma = sigma
        self.Delta = Delta
        self.expiry = expiry
        self.t = t
        self.lnc = lnc
        self.df = df

    def delta(self):
        d1 = (np.log(self.F / self.K) + (0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = (np.log(self.F / self.K) + (-0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
        return self.expiry * self.lnc - self.df * self.Delta * si.norm.cdf(-d2, 0.0, 1.0) - self.F * si.norm.cdf(-d1, 0.0, 1.0)

    def gamma(self):
        d1 = (np.log(self.F / self.K) + (0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
        return -self.expiry * self.delta() + self.Delta * self.df * si.norm.pdf(d1, 0.0, 1.0) / (self.F * self.sigma * np.sqrt(self.t)) - self.expiry * si.norm.cdf(d1, 0.0, 1.0)

    def vega(self):
        d1 = (np.log(self.F / self.K) + (0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
        return self.Delta * self.df * self.F * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(self.t)

    def theta(self):
        d1 = (np.log(self.F / self.K) + (0.5 * self.sigma ** 2) * (self.t + 1/12)) / (self.sigma * np.sqrt(self.t + 1/12))
        d2 = (np.log(self.F / self.K) + (-0.5 * self.sigma ** 2) * (self.t + 1/12)) / (self.sigma * np.sqrt(self.t + 1/12))
        d3 = (np.log(self.F / self.K) + (0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
        d4 = (np.log(self.F / self.K) + (-0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))

        df1 = 1/(1 + self.F) ** (self.t + 1/12)
        df2 = 1/(1 + self.F) ** self.t

        p1 = self.Delta * df1 * (self.K * si.norm.cdf(-d2, 0.0, 1.0) - self.F * si.norm.cdf(-d1, 0.0, 1.0))
        p2 = self.Delta * df2 * (self.K * si.norm.cdf(-d4, 0.0, 1.0) - self.F * si.norm.cdf(-d3, 0.0, 1.0))

        return (p1-p2) / (1/12)

class volatilityStripping(caplet):

    def __init__(self,K,F,sigma,delta,t,dt,start,length):
        self.K = K
        self.F = F
        self.sigma = sigma
        self.delta = delta
        self.t = t
        self.dt = dt
        self.start = start
        self.length = length # t + delta

    def caps(self):
        quarter = self.start
        qcaps = []
        for tt in range(0,5):
            for i in dt:
                if i*100 < 0.75*100:
                    model = caplet(self.K, self.F, self.sigma, self.delta, self.length+i, self.t+i).logNormalCaplet()
                    quarter += [model]
                    qcaps = np.cumsum(quarter)
                else:
                    model = caplet(self.K, self.F, self.sigma, self.delta, self.length + i, self.t + i).logNormalCaplet()
                    quarter += [model]
                    qcaps = np.cumsum(quarter)
            self.start = qcaps[-1]
            #print(f'{self.t},  {self.start}')
            self.t += 1
            if tt == 0:
                self.sigma = 0.2
            elif tt == 1:
                self.sigma = 0.225
            elif tt == 2:
                self.sigma = 0.225
            elif tt == 3:
                self.sigma = 0.25
            self.length = self.t + self.delta
        tt+=1

        capLst = qcaps
        return capLst

    def capIV(self, K, F, sigma, delta, length, t, curve):
        model = caplet(self.K, self.F, self.sigma, self.delta, self.length, self.t).logNormalCaplet()
        IV = root(lambda iv: (-(self.volHelper(K, F, iv, delta, length, t)) + curve),0.001)
        return IV.x

    def volHelper(self, K, F, sigma, delta, length, t):
        d1 = (np.log(F / K) + (0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = (np.log(F / K) + (-0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        df = 1/(1 + F)**length
        return delta * df * (K * si.norm.cdf(-d2, 0.0, 1.0) - F * si.norm.cdf(-d1, 0.0, 1.0))





if __name__ == '__main__':      # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # european option inputs
    S = 0.0125
    K = 0.0125
    T = 1.0
    r = 0.0
    sigma = 0.15

    Base = base()
    df = Base.discountFactor(r, T)
    print(f'{repr(Base)}\n')
    print(f'The call is:    {Base.euroCall(S, K, T, r, sigma)}')
    print(f'The put is:     {Base.euroPut(S, K, T, r, sigma)}')
    print(f'discount factor:   {Base.discountFactor(r, T)}\n')

    Greeks = euroGreeks(S, K, T, r, sigma)
    cDelta, pDelta = Greeks.delta()
    print(f'call delta:  {cDelta}')
    print(f'put delta:  {pDelta}')
    print(f'gamma:  {Greeks.gamma()}')
    print(f'vega:   {Greeks.vega()}')
    print(f'put theta:   {Greeks.theta()[1]}')
    print(f'call theta:   {Greeks.theta()[0]}')
    print('------------------tests above------------------\n')

    # Problem 1
    # caplet inputs
    K = 0.0125
    F = 0.0125
    sigma = 0.15
    delta = 0.25
    expiry = 1.25
    t = 1

    Caplet = caplet(K, F, sigma, delta, expiry, t)
    lnCaplet = Caplet.logNormalCaplet()
    bachCaplet = Caplet.bachelierCaplet()
    ivCaplet = Caplet.impliedVolatility(K, F, sigma, delta, expiry, t)
    print(f'Log Normal value:  {lnCaplet}')
    print(f'Bachelier value:    {bachCaplet}')
    print(f'Implied volatility:  {ivCaplet}\n')

    CG = capletGreeks(K, F, sigma, delta, expiry, t, lnCaplet, df)
    deltaCaplet = CG.delta()
    gammaCaplet = CG.gamma()
    vegaCaplet = CG.vega()
    thetaCaplet = CG.theta()
    print(f'delta:  {deltaCaplet}')
    print(f'gamma: {gammaCaplet}')
    print(f'vega:  {vegaCaplet}')
    print(f'theta:  {thetaCaplet}\n')

    K = [0.005,0.0075,0.01,0.0125]
    F = 0.0125
    sigma = 0.15
    delta = 0.25
    expiry = 1.25
    t = 1

    for k in K:
        print(f'-----Output for differing strikes, K = {k}-----')
        Caplet = caplet(k, F, sigma, delta, expiry, t)
        lnCaplet = Caplet.logNormalCaplet()
        bachCaplet = Caplet.bachelierCaplet()
        ivCaplet = Caplet.impliedVolatility(k, F, sigma, delta, expiry, t)
        print(f'Log Normal Put value:  {lnCaplet}')
        print(f' Bachelier Put value:  {bachCaplet}')
        print(f'  Implied volatility:  {ivCaplet}\n')

        CG = capletGreeks(k, F, sigma, delta, expiry, t, lnCaplet, df)
        deltaCaplet = CG.delta()
        gammaCaplet = CG.gamma()
        vegaCaplet = CG.vega()
        thetaCaplet = CG.theta()
        print(f'delta:  {deltaCaplet}')
        print(f'gamma:  {gammaCaplet}')
        print(f' vega:  {vegaCaplet}')
        print(f'theta:  {thetaCaplet}\n')





    # problem 2
    K = 0.01
    F = 0.01
    sigma = 0.15
    delta = 0.25
    start = [0]
    dt = [0, 0.25, 0.5, 0.75]
    t = 2
    length = t + delta

    vs = volatilityStripping(K,F,sigma,delta,t,dt,start,length)    # this needs work
    curve = vs.caps()
    capCurve = np.array([0.0002065, 0.00042493, 0.00065455, 0.00089472, 0.00122756, 0.00157298, 0.0019304, 0.0022993, 0.00272592, 0.00316435, 0.00361414, 0.00407486, 0.00454613, 0.00502759, 0.00551889, 0.00601973, 0.00658481, 0.00715975, 0.00774423, 0.008338])
    plt.plot(capCurve, label='Caplets', marker='*', color='b')
    plt.grid(linestyle='--', linewidth=1)
    plt.ylabel("yields")
    plt.xlabel("tenors in quarters")
    plt.title("Caplet Structure")
    plt.show()
    #print(capCurve)

    impliedVolatility = []
    j = 0.0
    m= 1
    for i in range(0,len(capCurve)):
        if i > 0:
            impliedVolatility += [float(vs.capIV(K,F,sigma,delta*m,2+j,t,capCurve[i]))]
        else:
            impliedVolatility += [float(vs.capIV(K, F, sigma, delta, 2.25, t, capCurve[i]))]
        j+=0.25
        m+=1
    #print(impliedVolatility)

    marketVol = np.array([0.15, 0.15, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.25, 0.25, 0.25, 0.25])
    plt.plot(impliedVolatility, label='Stripped Vol', marker='*', color='b')
    plt.plot(marketVol, label='Realized Vol', marker='o', color='r')
    plt.grid(linestyle='--', linewidth=1)
    plt.ylabel("Implied Vol.")
    plt.xlabel("Tenors in Quarters")
    plt.title("Volatility Structures")
    plt.legend()
    plt.show()
    #print(capCurve)