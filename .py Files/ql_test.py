import QuantLib as ql

print('THis is all just to test quantLib')
strike_price = 110.0
payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)

maturity = ql.Date(15, 1, 2016)
S = 127.62
K = 130
Vol = 0.20
q = 0.0163
type = ql.Option.Call
r = 0.001
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()

calc_date = ql.Date(8, 5, 2015)
ql.Settings.instance().evaluationDate = calc_date

payoff = ql.PlainVanillaPayoff(type, S)
exercise = ql.EuropeanExercise(maturity)
euro_option = ql.VanillaOption(payoff, exercise)

v0 = Vol*Vol
kappa = 0.1
theta = v0
sigma = 0.1
rho = -0.75
spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, r, day_count))
div = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, q, day_count))
heston = ql.HestonProcess(flat_ts, div, spot_handle, v0, kappa, theta, sigma, rho)

engine = ql.AnalyticHestonEngine(ql.HestonModel(heston), 0.01, 1000)
euro_option.setPricingEngine(engine)
h_price = euro_option.NPV()
print(f'The Heston model price:  {h_price}')
