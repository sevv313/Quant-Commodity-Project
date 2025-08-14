import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json
import re
import databento as db
from datetime import datetime, timedelta
from scipy.interpolate import make_interp_spline
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline 


# Read data
df_0 = pd.read_csv('data/ICE-past-1-year/BGDb.csv')
df_1 = pd.read_csv('data/CME-past-5-year/CME.csv')
df_2 = pd.read_csv('data/CME-WTI-past-1-year/WTI.csv')

# Filter outright contracts
def filter_outright(df, product_prefixes=None):
    if product_prefixes is None:
        product_prefixes = []
    mask = (~df['symbol'].astype(str).str.contains('[-_]', na=False)) & \
           (df['close'].notna()) & (df['close'] != 0)
    if product_prefixes:
        mask &= df['symbol'].astype(str).str.startswith(tuple(product_prefixes))
    return df[mask].copy()

#Extract ICE contract info
def extract_ice_contract_info(symbol):
    s = str(symbol)
    m = re.search(r'FM([FGHJKMNQUVXZ])(\d{4})', s)
    if not m:
        return None, None
    month_code, year4 = m.groups()
    month_map = {'F':1,'G':2,'H':3,'J':4,'K':5,'M':6,'N':7,'Q':8,'U':9,'V':10,'X':11,'Z':12}
    yy = int(year4[-2:])
    year_full = 2000 + yy
    return pd.Timestamp(year_full, month_map[month_code], 15), f"{month_code}{yy:02d}"

#Extract CME contract info
def extract_cme_contract_info(symbol):
    s = str(symbol)
    if '-' in s:  # skip spreads
        return None, None
    m = re.match(r'([A-Z]{2})([FGHJKMNQUVXZ])(\d)', s)
    if not m:
        return None, None
    product, month_code, year_digit = m.groups()
    month_map = {'F':1,'G':2,'H':3,'J':4,'K':5,'M':6,'N':7,'Q':8,'U':9,'V':10,'X':11,'Z':12}
    month = month_map[month_code]
    year_full = 2020 + int(year_digit)
    return pd.Timestamp(year_full, month, 15), f"{month_code}{year_digit}"

#Product identifyer#
def identify_product(symbol):
    if isinstance(symbol, str):
        if symbol.startswith("G"):
            return "Gasoil"
        elif symbol.startswith("BRN"):
            return "Brent"
        elif symbol.startswith("DBI"):
            return "Dubai"
        elif symbol.startswith("HO"):
            return "Heating Oil"
        elif symbol.startswith("RB"):
            return "Gasoline"
        elif symbol.startswith(("HO","RB","CL","NG","ZL")):  # CME examples
            return symbol[:2]  # simple product code
    return "Unknown"


#Parse function
def parse_contracts(df, extract_func):
    results = df['symbol'].apply(lambda s: extract_func(s) or (pd.NaT, None))
    df['expiration'], df['contract_code'] = zip(*results)
    df['product_type'] = df['symbol'].apply(identify_product)
    return df[df['expiration'].notna()].copy()

#Converting Price Unit
def adjust_prices(df):
    df.loc[df['product_type'] == 'Gasoil', 'close'] /= 7.45
    df.loc[df['product_type'] == 'Heating Oil', 'close'] *= 42
    df.loc[df['product_type'] == 'Gasoline', 'close'] *= 42
    
    return df

#Process ICE
outright_only0 = filter_outright(df_0)
outright_only0 = parse_contracts(outright_only0, extract_ice_contract_info)
forward_curve_data0 = adjust_prices(outright_only0)

#Process CME
outright_only1 = filter_outright(df_1, product_prefixes=[])  # optional prefix filter
outright_only1 = parse_contracts(outright_only1, extract_cme_contract_info)
forward_curve_data1 = adjust_prices(outright_only1)

#Process CME-WTI
outright_only2 = filter_outright(df_2, product_prefixes=[])  # optional prefix filter
outright_only2 = parse_contracts(outright_only2, extract_ice_contract_info)
forward_curve_data2 = adjust_prices(outright_only2)

print("ICE data")
print(forward_curve_data0[['ts_event', 'symbol', 'expiration', 'close', 'product_type']].head())
print("\nCME data")
print(forward_curve_data1[['ts_event', 'symbol', 'expiration', 'close', 'product_type']].head())
print("\nWTI data")
print(forward_curve_data2[['ts_event', 'symbol', 'expiration', 'close', 'product_type']].head())

#Curve Plotting# #
# def get_forward_curve(df, product_type, target_dates):
#     df["ts_event"] = pd.to_datetime(df["ts_event"])
#     target_dates = [pd.to_datetime(d).date() for d in target_dates]
#     curves = {}
#     for date in target_dates:
#         df_filtered = df[df["product_type"].str.contains(product_type, case=False, na=False) & (df["ts_event"].dt.date == date)].copy()
#         df_filtered = df_filtered.sort_values("expiration")
#         curves[date] = df_filtered
#     return curves

# def plot_forward_curves(df, products, target_dates):
#     for product in products:
#         curves = get_forward_curve(df, product, target_dates)
#         plt.figure(figsize=(10, 6))
#         plotted_any = False
        
#         for date, df_curve in curves.items():
#             if df_curve.empty:
#                 continue
            
#             df_curve = df_curve.drop_duplicates(subset="expiration")
#             x = (df_curve["expiration"] - df_curve["expiration"].min()).dt.days
#             y = df_curve["close"]
            
#             if len(x) >= 2:
#                 cs = CubicSpline(x, y)
#                 x_smooth = np.linspace(x.min(), x.max(), 300)
#                 y_smooth = cs(x_smooth)
#                 dates_smooth = df_curve["expiration"].min() + pd.to_timedelta(x_smooth, unit='D')
#                 plt.plot(dates_smooth, y_smooth, label=str(date))
#             # fallback: plot raw points
#             plt.scatter(df_curve["expiration"], df_curve["close"], color='red')
#             plotted_any = True
        
#         if plotted_any:
#             plt.title(f"{product} Forward Curves - Cubic Spline")
#             plt.xlabel("Expiration Date")
#             plt.ylabel("Close Price")
#             plt.grid(True)
#             plt.xticks(rotation=45)
#             plt.legend()
#             plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))  # e.g., Jan-25
#             plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())         # automatically pick ticks
#             plt.xticks(rotation=45)
#             plt.show()
#         else:
#             print(f"No data found for product: {product} on target dates.")

# # Input target dates and products
# target_dates = ["2025-07-11", "2025-08-11", "2025-02-11", "2024-08-12"]
# products = ["Brent", "Dubai"]

# plot_forward_curves(outright_only, products, target_dates)




#Plot ICE together#
# def plot_all_products_together(df, products, target_dates):
#     plt.figure(figsize=(12, 6))
    
#     # Assign a distinct color for each product
#     product_colors = {
#         "Gasoil": "blue",
#         "Brent": "green",
#         "Dubai": "orange"
#     }
    
#     # Line styles for different target dates
#     line_styles = ['-', '--', '-.', ':']
    
#     df["ts_event"] = pd.to_datetime(df["ts_event"])
#     target_dates_parsed = [pd.to_datetime(d).date() for d in target_dates]
    
#     for i, product in enumerate(products):
#         color = product_colors.get(product, "black")
        
#         for j, date in enumerate(target_dates_parsed):
#             style = line_styles[j % len(line_styles)]
            
#             df_curve = df[df["product_type"].str.contains(product, case=False, na=False) & 
#                           (df["ts_event"].dt.date == date)].copy()
#             if df_curve.empty:
#                 continue
            
#             df_curve = df_curve.sort_values("expiration").drop_duplicates(subset="expiration")
            
#             x = (df_curve["expiration"] - df_curve["expiration"].min()).dt.days
#             y = df_curve["close"]
            
#             # Keep only finite values
#             mask = np.isfinite(x) & np.isfinite(y)
#             x = x[mask].to_numpy()
#             y = y[mask].to_numpy()
            
#             # Ensure strictly increasing x
#             _, idx = np.unique(x, return_index=True)
#             x = x[idx]
#             y = y[idx]
            
#             if len(x) >= 2:
#                 cs = CubicSpline(x, y)
#                 x_smooth = np.linspace(x.min(), x.max(), 300)
#                 y_smooth = cs(x_smooth)
#                 dates_smooth = df_curve["expiration"].min() + pd.to_timedelta(x_smooth, unit='D')
#                 plt.plot(dates_smooth, y_smooth, color=color, linestyle=style, 
#                          label=f"{product} - {date}")
            
#             # plot raw points as fallback
#             plt.scatter(df_curve["expiration"], y, color=color, edgecolor='black', marker='o')
    
#     # Format x-axis as Month-Year
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
#     plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
#     plt.xticks(rotation=45)
    
#     plt.title("Forward Curves for Gasoil, Brent, and Dubai")
#     plt.xlabel("Expiration Date")
#     plt.ylabel("Close Price")
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# target_dates = ["2025-07-11", "2025-08-11", "2025-02-11", "2024-08-12"]
# products = ["Gasoil", "Brent", "Dubai"]

# plot_all_products_together(outright_only0, products, target_dates)



# x = df_curve["expiration"].map(datetime.toordinal) 
# y = df_curve["close"] 
# spline = UnivariateSpline(x, y, s=0) 
# xs = np.linspace(x.min(), x.max(), 200)