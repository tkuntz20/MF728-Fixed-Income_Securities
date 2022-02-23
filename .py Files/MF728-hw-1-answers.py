# -*- coding: utf-8 -*-
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


def dayCountConvention(start,maturity,period,choice):
    start = datetime.strptime(start, "%d,%m,%Y")
    maturity = datetime.strptime(maturity,"%d,%m,%Y")
    if choice == "Actual/Actual":
        quant = ql.ActualActual()
        convention = abs(maturity - start).days
        year = start.year
        year = 366 if calendar.isleap(year) else 365
        return quant, convention / year
    elif choice == "Actual/365":
        quant = ql.Actual365Fixed()
        convention = abs(maturity - start).days
        return quant, convention / 365
    elif choice == "Actual/360":
        quant = ql.Actual360()
        convention = abs(maturity - start).days
        return quant, convention / 360
    elif choice == "Business252":
        quant = ql.Business252()
        convention = abs(maturity - start).days
        return quant, convention / 252
    elif choice == "30/360":
        quant = ql.Thirty360()
        convention = 30
        return quant, convention / 360
    else:
        return "A valid Daycount Convention was not passed in."

def discountFactor(years, facevalue, YTM, periods):
    return facevalue / np.power((1 + (YTM / periods)), years * periods)

def survivalProbabilityCurve(HazardRates,recoveryRate, S, maturities):
    P1 = [None]*len(maturities)                                     # P1 is the survival probability
    P1[0] = np.exp(-HazardRates[0] * maturities[0])
    #print('time 0 survival:   ', P1[0])
    
    for i in range(1, len(maturities)):
        x = (maturities[i] - maturities[i-1])
        P1[i] = round(np.exp(np.log(P1[i-1]) - (HazardRates[i-1] * x)),8)
    #print('time 0-t survival: ',P1)
    P1.insert(0, 1)
    return P1

def hazardRateCurve(recoveryRate, S, maturities):
    hazVector = [None]*len(maturities)
    #print('hazard vector:  ',hazVector)
    for i in range(0, len(maturities)):
        hazVector[i] = round((S[i] / 10000) / (1 - recoveryRate),8)
    print('hazard vector filled:  ',hazVector)
    return hazVector
    
def PremiumLeg(maturities, S, recoveryRate, survivProb, hazRate, years, facevalue, YTM, periods, dummy):
    
    riskAnnuity = 0
    premium = 0
    N = int(periods*maturities)
    #print('N is  ',N)
    for n in range(1, N+1):
        #print('n is ->',n)
        tn = n / periods
        tn_1 = (n-1) / periods
        dt = 1.0 / periods
        #print('survivProb ->',survivProb)
        #print('survivProb[n-1] ->',survivProb[n-1])
        riskAnnuity += dt * discountFactor(years, facevalue, YTM, tn) * survivProb[n-1]
        #print('risky annuity -> ',riskAnnuity)
        premium += 0.5 * dt * discountFactor(years, facevalue, YTM, tn) * (survivProb[n-2] + survivProb[n-1])
        #print('premium -> ', premium) 
    if dummy == 'Y':
        #print('with accruel ->',S * (riskAnnuity + premium))
        return S * (riskAnnuity + premium)
    else:
        return S * riskAnnuity

def contingentLeg(maturities, S, recoveryRate, survivProb, hazRate, years, facevalue, YTM, periods):
    
    riskAnnuity = 0
    contingent = 0
    N = int(periods*maturities)
    #print('N is  ',N)
    for n in range(1, N+1):
        #print('n is ->',n)
        tn = n / periods
        tn_1 = (n-1) / periods
        dt = 1.0 / periods
        #print('survivProb[n] ->',survivProb[n])
        #print('survivProb[n-1] ->',survivProb[n-1])
        riskAnnuity += discountFactor(years, facevalue, YTM, tn) * (survivProb[n-2] - survivProb[n-1])
        #print('risky annuity -> ',riskAnnuity)
    return (1-recoveryRate) * riskAnnuity


def midPoint(survivRate1,survivRate2,maturity1,maturity2,recoveryRate):
    
    fourYrSV = (survivRate1 + survivRate2) / 2  #midpoint approx of the four year survival rate 
    print('mid poit, ', fourYrSV)
    s1 = round((fourYrSV * 100),0)              #rounds the values to zero decimal places
    s2 = round((survivRate2 *100),0)
    mp = (np.log(s1) - np.log(s2)) / (maturity2-maturity1)
    
    return mp * (1 - recoveryRate), s1
    
def markToMarket(maturities, fourYrpremium, spread, survivProb,fourYrsurviv):
        
    probVector = survivProb
    probVector.insert(-1,fourYrsurviv/100)
    probVector.pop(0)
    #print(probVector)
    rpv01 = 0
    DF = []
    for i in range(1,len(probVector)+1):
        DF.append(discountFactor(i, 1, 0.02, 1))
    DF.pop(0)
    #print(DF)
    for i in range(0,len(maturities)):
        rpv01 += DF[i] * (probVector[i] + probVector[i+1])
    return (0.5 * rpv01) * (fourYrpremium - spread)/10000

def dollarBill(survivProb,fourYrsurviv):
    
    probVector = survivProb
    probVector.insert(-1,fourYrsurviv/100)
    probVector.pop(0)
    DF = []
    for i in range(0,5):
        DF.append(discountFactor(i, 1, 0.03, 1))
    #print(DF)
    dv1 = 50 * DF[0] * (1 + probVector[0])
    dv2 = 55 * ((DF[0] * (1 + probVector[0])) + (DF[1] * (probVector[0] + probVector[1])))
    value = DF[0] * (1 + probVector[0])
    return value / 2
    
def curveDollarBill(survivProb,fourYrsurviv):
    probVector = survivProb
    probVector.insert(-1,fourYrsurviv/100)
    probVector.pop(0)
    DF = []
    for i in range(0,5):
        DF.append(discountFactor(i, 1, 0.03, 1))  
    return 50 * (((DF[0] + 0.0001) * (1 + probVector[0])) - (DF[0] * (1 + probVector[0])))

def dollarBillsensitivity(survivProb,fourYrsurviv):
    probVector = survivProb
    probVector.insert(-1,fourYrsurviv/100)
    probVector.pop(0)
    DF = []
    for i in range(0,5):
        DF.append(discountFactor(i, 1, 0.02, 1))  
    return 0.01 * (DF[0] * (1 - probVector[0]))

if __name__ == '__main__':      # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    

    q, convention = dayCountConvention('7,2,2022', '7,3,2023',ql.Period('1Y'), 'Actual/Actual')
    print(convention)

    hazardVec = hazardRateCurve(0.4, [100,110,120,140],[1,2,3,5])
    print('hazard rates ->',hazardVec)
    
    survivProb = survivalProbabilityCurve(hazardVec, 0.4, [100/10000,110/10000,120/10000,140/10000],[1,2,3,5])
    print('survival probs  ', survivProb)
    
    DF = discountFactor(1, 100, 0.02, 1)
    print('discount factor:  ',DF,'\n')
    
    # premLeg1 = PremiumLeg(1, 100/10000, 0.4, survivProb, hazardVec, 1, 1, 0.02, 1, 'Y')
    # print('1-yr premium leg is  ',premLeg1)
    # contLeg1 = contingentLeg(1, 100/10000, 0.4, survivProb, hazardVec, 1, 1, 0.02, 1)
    # print('1-yr contingent leg is  ',contLeg1,'\n')
    
    # premLeg2 = PremiumLeg(2, 110/10000, 0.4, survivProb, hazardVec, 2, 1, 0.02, 1, 'Y')
    # print('2-yr premium leg is  ',premLeg2)
    # contLeg2 = contingentLeg(2, 110/10000, 0.4, survivProb, hazardVec, 2, 1, 0.02, 1)
    # print('2-yr contingent leg is  ',contLeg2,'\n')
    
    # premLeg3 = PremiumLeg(3, 120/10000, 0.4, survivProb, hazardVec, 3, 1, 0.02, 1, 'Y')
    # print('3-yr premium leg is  ',premLeg3)
    # contLeg3 = contingentLeg(3, 120/10000, 0.4, survivProb, hazardVec, 3, 1, 0.02, 1)
    # print('3-yr contingent leg is  ',contLeg3,'\n')
    
    # premLeg5 = PremiumLeg(5, 140/10000, 0.4, survivProb, hazardVec, 5, 1, 0.02, 1, 'Y')
    # print('5-yr premium leg is  ',premLeg5)
    # contLeg5 = contingentLeg(5, 140/10000, 0.4, survivProb, hazardVec, 5, 1, 0.02, 1)
    # print('5-yr contingent leg is  ',contLeg5,'\n')
    
    midpoint = midPoint(survivProb[-2], survivProb[-1], 4, 5, 0.4)
    print('the midpoint is ->',midpoint[0])
    
    mtm = markToMarket([1,2,3,5],midpoint[0], 80, survivProb,midpoint[1])
    print('5-yr mark-to-market value ->',mtm)
    
    
    print(dollarBill(survivProb,midpoint[1]))
    
    print(curveDollarBill(survivProb,midpoint[1]))
    
    print(dollarBillsensitivity(survivProb,midpoint[1]))
    # cdsCurve = [premLeg1-contLeg1,premLeg2-contLeg2,premLeg3-contLeg3,premLeg5-contLeg5]
    # plt.title('CDS Hazard spreads', fontsize = 15)
    # plt.xlabel('Time to Maturity (in Years)',fontsize = 10)
    # plt.ylabel('Yields in Basis Points',fontsize = 10)
    # plt.grid(linestyle = '--', linewidth = 1)
    # plt.plot([1,2,3,5],cdsCurve,scalex=(True),scaley=(True),marker='o')
    
    