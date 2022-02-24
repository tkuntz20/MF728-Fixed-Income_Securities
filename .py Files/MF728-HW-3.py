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

    #def __repr__(self):
    #    return f'initial level is {self.S}, strike is {self.K}, expiry is {self.T}, interest rate is {self.r}, volatility is {self.sigma}.'

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

    #def zeroCoupon(self, years, facevalue, YTM, periods):
    #    return facevalue / np.power((1 + (YTM / periods)), years * periods)

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
        d1 = (np.log(self.F / self.K) + (0.5 * self.sigma ** 2) * self.expiry) / (self.sigma * np.sqrt(self.t))
        d2 = (np.log(self.F / self.K) + (-0.5 * self.sigma ** 2) * self.expiry) / (self.sigma * np.sqrt(self.t))

        return self.delta * self.discountFactor(self.F, self.expiry) * (self.K * si.norm.cdf(-d2, 0.0, 1.0) - self.F * si.norm.cdf(-d1, 0.0, 1.0))

    def bachelierCaplet(self):
        d = (self.F - self.K) / (self.sigma * np.sqrt(self.t))
        return self.delta * self.sigma * self.discountFactor(self.F,self.expiry) * np.sqrt(self.t) * (-d * si.norm.cdf(-d, 0.0, 1.0) + si.norm.pdf(-d, 0.0, 1.0))

    def volHelper(self,K,F,sigma,delta,expiry,t):
        d = (F - K) / (sigma * np.sqrt(t))
        return delta * sigma * self.discountFactor(F, expiry) * np.sqrt(t) * (-d * si.norm.cdf(-d, 0.0, 1.0) + si.norm.pdf(-d, 0.0, 1.0))

    def impliedVolatility(self,K,F,sigma,delta,expiry,t):
        IV = root(lambda iv: np.abs((self.volHelper(K,F,iv,delta,expiry,t))/100 - self.logNormalCaplet()),0.001)
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
    print(f'Log Normal value:  {lnCaplet}\n')
    print(f'Bachelier value:    {bachCaplet}\n')
    print(f'Implied volatility:  {ivCaplet}\n')

    CG = capletGreeks(K, F, sigma, delta, expiry, t, lnCaplet, df)
    deltaCaplet = CG.delta()
    gammaCaplet = CG.gamma()
    vegaCaplet = CG.vega()
    thetaCaplet = CG.theta()
    print(f'delta:  {deltaCaplet}')
    print(f'gamma: {gammaCaplet}')
    print(f'vega:  {vegaCaplet}')
    print(f'theta:  {thetaCaplet}')
