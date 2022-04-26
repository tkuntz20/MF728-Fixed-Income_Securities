# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 21:55:09 2022

@author: Thomas Kuntz MF728-HW-2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class IRStransforms():                                                         # class to extract FRAs, zeros and discount factors

    def __init__(self, swapCurve, maturities, timeDelta):
        # swapCurve is the market observed swap curve
        # maturities is the simple the matrities of the swaps in swapCurve
        # timeDelta is the change in time in years between the maturities
        self.swapCurve = swapCurve
        self.maturities = maturities
        self.timeDelta = timeDelta
        
    def __repr__(self):

        sumar = pd.DataFrame([self.maturities,self.swapCurve,self.timeDelta]).T
        sumar.columns = ['Tenors','Swaps','Deltas']
        return f' {sumar}\n'
        
    def forwardRateExtractor(self):                                            # pulls implied forwards from market observed swaps
        forwardCurve = np.zeros(len(self.swapCurve))
        forwardCurve[0] = 1 / (1 + self.swapCurve[0] * 1)
        for n in range(1, len(self.swapCurve)):
            FRA = (n+1) - sum(forwardCurve[:n]) - self.swapCurve[n] * np.sum(self.timeDelta[:n] * forwardCurve[:n])
            forwardCurve[n] = FRA / (1 + self.swapCurve[n] * self.timeDelta[n])
        forward = np.array([self.FRAFromDiscount(forwardCurve[t], self.timeDelta[t]) for t in range(len(self.swapCurve))])
        return forward, forwardCurve

    def FRAFromDiscount(self, discountFactor, timeDelta):                      # helper func. extracts a FRA from derived or given discount factors 
        return 1 / timeDelta * (1/discountFactor-1)

    def discountFromFRA(self, forward, timeDelta):                             # helper func. extracts discounts from a given set of fowards
        return 1 / (forward * timeDelta + 1)

    def shiftedSwaps(self, forwardRates, discountFactors):                     # calculates a set of swaps from forwards and disc. factors 
        forwardRates = np.array(forwardRates) + 0.01
        annuity = 0
        floatingLeg = 0
        discountFactors = np.array([self.discountFromFRA(forwardRates[i], self.timeDelta[i]) for i in range(len(forwardRates))])
        swapCurve = np.zeros(len(forwardRates))
        #print(f'calc d disc. facts. :  {discountFactors}\n')
        #print(f'shifted forwards curve :  {forwardRates}\n')
        for n in range(len(forwardRates)):
            floatingLeg += self.timeDelta[n] * forwardRates[n] * discountFactors[n]
            annuity += self.timeDelta[n] * discountFactors[n]
            swapCurve[n] = floatingLeg / annuity
            #swapCurve.append(floatingLeg / annuity)
        return swapCurve, discountFactors, forwardRates
    
    def fifteenYrSwap(self,forwardRates, discountFactors):
        delta15Yr = self.timeDelta
        delta15Yr[-1] = 15
        #print(delta15Yr)
        fixedLeg = np.sum(forwardRates * discountFactors * delta15Yr)
        annuity = np.sum(discountFactors * delta15Yr)
        return fixedLeg / annuity
    

if __name__ == '__main__':      # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    

    swapCurve = [0.028438,0.03060,0.03126,0.03144,0.03150,0.03169,0.03210,0.03237]
    maturities = [1,2,3,4,5,7,10,30]
    timeDelta = [1,1,1,1,1,2,3,20]
    IRS = IRStransforms(swapCurve, maturities, timeDelta)
    print(' Initialization of curves \n')
    print(repr(IRS))
    
    # part a
    forwardRates, discountFactors = IRS.forwardRateExtractor()
    print('Tenors | Forwards | Disc. Factors')
    for i in range(len(forwardRates)):
        if i < 6:
            print(f'  {maturities[i]}    | {np.round(forwardRates[i],5)}  |   {np.round(discountFactors[i],5)}')
        else:
            print(f'  {maturities[i]}   | {np.round(forwardRates[i],5)}  |   {np.round(discountFactors[i],5)}')
    print()
    
    #part b
    interpolatedSwap = IRS.fifteenYrSwap(forwardRates, discountFactors)
    print(f'Break Even 15yr swap rate is:  {interpolatedSwap}\n')
    
    # part c
    
    # part d
    
    # part e
    zeroRates = [-np.log(discountFactors[i]) / timeDelta[i] for i in range(len(timeDelta))]
    print('              Implied Rates Table')
    print('Tenors | Forwards | Disc. Factors | Zero Rates')
    for i in range(len(forwardRates)):
        if i < 6:
            print(f'  {maturities[i]}    | {np.round(forwardRates[i],5)}  |   {np.round(discountFactors[i],5)}     |  {np.round(zeroRates[i],5)}')
        else:
            print(f'  {maturities[i]}   | {np.round(forwardRates[i],5)}  |   {np.round(discountFactors[i],5)}     |  {np.round(zeroRates[i],5)}')
    print()
    
    # part f
    newSwaps, newDiscFacts, newForwards = IRS.shiftedSwaps(forwardRates, discountFactors)
    print('          Shifted Curves (+100bps)')
    print('Tenors |  Swaps  |  Forwards  | Disc. Factors')
    for i in range(len(forwardRates)):
        if i < 6:
            print(f'  {maturities[i]}    | {np.round(newSwaps[i],4)}  |   {np.round(newForwards[i],5)}  |   {np.round(newDiscFacts[i],5)}')
        else:
            print(f'  {maturities[i]}   | {np.round(newSwaps[i],4)}  |   {np.round(newForwards[i],5)}  |   {np.round(newDiscFacts[i],5)}')
    print()
    
    # part g
    bearShift = np.array([0,0,0,0.0005,0.001,0.0015,0.0025,0.005])
    bearSwapCurve = np.ndarray.tolist(np.array(swapCurve) + bearShift)
    bearswapDF = pd.DataFrame(bearSwapCurve,maturities)

    # part h
    IRSbearish = IRStransforms(bearSwapCurve, maturities, timeDelta)
    bearishForwards, bearishDiscountFactors = IRSbearish.forwardRateExtractor()
    print('           Bearish Curve Shifts')
    print('Tenors |  Swaps  |  Forwards  | Disc. Factors')
    for i in range(len(forwardRates)):
        if i < 6:
            print(f'  {maturities[i]}    | {np.round(bearSwapCurve[i],4)}  |   {np.round(bearishForwards[i],5)}  |   {np.round(bearishDiscountFactors[i],5)}')
        else:
            print(f'  {maturities[i]}   | {np.round(bearSwapCurve[i],4)}  |   {np.round(bearishForwards[i],5)}  |   {np.round(bearishDiscountFactors[i],5)}')
    print()
    
    # part i
    bullShift = np.array([-0.005,-0.0025,-0.0015,-0.001,-0.0005,0.0,0.0,0.0])
    bullSwapCurve = np.ndarray.tolist(np.array(swapCurve) + bullShift)

    # part j
    IRSbullish = IRStransforms(bullSwapCurve, maturities, timeDelta)
    bullishForwards, bullishDiscountFactors = IRSbullish.forwardRateExtractor()
    print('           Bullish Curve Shifts')
    print('Tenors |  Swaps  |  Forwards  | Disc. Factors')
    for i in range(len(forwardRates)):
        if i < 6:
            print(f'  {maturities[i]}    | {np.round(bullSwapCurve[i],4)}  |   {np.round(bullishForwards[i],5)}  |   {np.round(bullishDiscountFactors[i],5)}')
        else:
            print(f'  {maturities[i]}   | {np.round(bullSwapCurve[i],4)}  |   {np.round(bullishForwards[i],5)}  |   {np.round(bullishDiscountFactors[i],5)}')

    # graphing
    df = pd.DataFrame([swapCurve,maturities,timeDelta,np.ndarray.tolist(newSwaps),bearSwapCurve,bearishForwards,bullSwapCurve, bullishForwards])
    df.T.head(9)

    plt.plot(forwardRates,label='Market Forwards',marker='>')
    plt.plot(bearishForwards,label='Bear Forwards',marker='o') 
    plt.plot(bullishForwards,label='Bull Forwards',marker='x')
    plt.plot(zeroRates,label='Market derived Zeros',marker='*')
    plt.grid(linestyle = '--', linewidth = 1)
    plt.ylabel("yields")
    plt.xlabel("tenors")
    plt.title("Forward Curves")
    default_x_ticks = range(len(maturities))
    plt.xticks(default_x_ticks, maturities)
    plt.legend()
    plt.show()

    plt.plot(discountFactors,label='Market DFs',marker='x')
    plt.plot(bearishDiscountFactors,label='Bear DFs',marker='o') 
    plt.plot(bullishDiscountFactors,label='Bull DFs',marker='>')
    plt.grid(linestyle = '--', linewidth = 1)
    plt.ylabel("yields")
    plt.xlabel("tenors")
    plt.title("Disc. Factor Curves")
    default_x_ticks = range(len(maturities))
    plt.xticks(default_x_ticks, maturities)
    plt.legend()
    plt.show()
    
    plt.plot(zeroRates,label='Market Derived Zeros',marker='x')
    plt.grid(linestyle = '--', linewidth = 1)
    plt.ylabel("yields")
    plt.xlabel("tenors")
    plt.title("Zeros Curve")
    default_x_ticks = range(len(maturities))
    plt.xticks(default_x_ticks, maturities)
    plt.legend()
    plt.show()

    plt.plot(swapCurve,label='Market Swaps',marker='x')
    plt.plot(bearSwapCurve,label='Bear Swaps',marker='o') 
    plt.plot(bullSwapCurve,label='Bull Swaps',marker='>')
    plt.grid(linestyle = '--', linewidth = 1)
    plt.ylabel("yields")
    plt.xlabel("tenors")
    plt.title("Swap Curves")
    default_x_ticks = range(len(maturities))
    plt.xticks(default_x_ticks, maturities)
    plt.legend()
    plt.show()

    print('test test test')