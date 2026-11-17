"""Live Google Sheets parsing, last-five-per-product normal parameter fitting.

No special-case refresh cache: every Run Simulation fetches the published CSV.
"""
import hashlib
import io
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from scipy.stats import truncnorm
from config import PRODUCTS, STAGES, STAGE_RESOURCES, STAGE_OBS, FMS_SHARES, SHEET_URL

COLUMN_ALIASES = {
 'order id':'order_id','order_id':'order_id','product':'product', 'product id':'product',
 'ws1':'WS1','ws2':'WS2','ws3':'WS3','ws4':'WS4','ws8':'WS8',
 'ws 5, 6, 7':'WS5-7','ws5–7 combined':'WS5-7','ws5-7 combined':'WS5-7',
 'ws5 6 7':'WS5-7', 'ws5–7':'WS5-7', 'ws5-7':'WS5-7',
}
NUM_COLS=['WS1','WS2','WS3','WS4','WS5-7','WS8']

def normalize_header(name):
    s=str(name).strip().lower().replace('(s)','').replace(' (s)','').replace('_',' ')
    s=' '.join(s.split())
    return COLUMN_ALIASES.get(s,s)

def parse_data(csv_text):
    df=pd.read_csv(io.StringIO(csv_text),dtype=str).dropna(how='all')
    df.columns=[normalize_header(x) for x in df.columns]
    if df.columns.duplicated().any(): raise ValueError('Duplicate normalized column names')
    required=['order_id','product',*NUM_COLS]
    missing=[x for x in required if x not in df.columns]
    if missing: raise ValueError(f'Missing sheet columns: {missing}; received {list(df.columns)}')
    df=df[required].copy()
    df['order_id']=df['order_id'].fillna('').str.strip()
    df['product']=df['product'].fillna('').str.strip().str.upper()
    # Order IDs are opaque identifiers, not required to contain valid dates.
    df=df[df['order_id'].ne('')].copy()
    df=df[df['product'].isin(PRODUCTS)].copy()
    for c in NUM_COLS: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.loc[(df[NUM_COLS]>0).all(axis=1)].copy()
    if df['order_id'].duplicated().any():
        raise ValueError('Duplicate valid order IDs; fix the sheet before simulation')
    # Chronology is the top-to-bottom order of observations in the Google Sheet.
    # New observations must be appended at the bottom; never sort by ID.
    df=df.reset_index(drop=True)
    df['row_sequence']=np.arange(1, len(df)+1)
    for p in PRODUCTS:
        if (df['product']==p).sum()<5: raise ValueError(f'{p} has fewer than five valid historical rows')
    return df

def fetch_live(url=SHEET_URL, timeout=25):
    response=requests.get(url,params={'_fresh':datetime.now(timezone.utc).timestamp()},
        headers={'Cache-Control':'no-cache','Pragma':'no-cache'},timeout=timeout)
    response.raise_for_status()
    raw=response.content.decode('utf-8-sig')
    if raw.lstrip().lower().startswith('<!doctype html') or raw.lstrip().lower().startswith('<html'):
        raise ValueError('The URL returned HTML, not a published CSV')
    return parse_data(raw),hashlib.sha256(raw.encode()).hexdigest()[:12]

def most_recent_five(df):
    return (df.groupby('product',group_keys=False).tail(5)
            .sort_values(['product','row_sequence']).copy())

def fit_normal(values):
    v=np.asarray(values,dtype=float)
    if len(v)<2 or not np.isfinite(v).all() or (v<=0).any():
        raise ValueError('Normal calibration needs at least 2 valid positive durations')
    return float(v.mean()), float(v.std(ddof=1))

def calibrate(df):
    latest=most_recent_five(df)
    rows=[]; observations={}
    for product in PRODUCTS:
        five=latest.loc[latest['product']==product]
        for stage in STAGES:
            column=STAGE_OBS[stage]
            share=FMS_SHARES[stage] if stage in FMS_SHARES else 1.0
            vals=(five[column].to_numpy(dtype=float)*share)
            mu, sd=fit_normal(vals)
            for machine in STAGE_RESOURCES[stage]:
                rows.append(dict(Resource=machine,Workstation=stage,Product=product,
                    **{'Mean (s)':mu,'SD (s)':sd,'N observations':len(vals)},
                    Distribution='Positive-truncated Normal',
                    Provenance=('Allocated WS5–WS7' if stage in FMS_SHARES
                      else 'Workstation → machine assumption' if len(STAGE_RESOURCES[stage])==2
                      else 'Observed workstation')))
                observations[(product,machine)]=(mu,sd)
    return latest,pd.DataFrame(rows),observations

def rolling_validation(df):
    """Out-of-sample checks of fitted duration distributions, NOT a forecast model.

    Each 5-record window fits a positive-truncated Normal. The next historical
    duration is held out and checked against its fitted distribution.
    """
    rows=[]
    for product in PRODUCTS:
        d=df[df['product']==product].sort_values('row_sequence')
        for column in NUM_COLS:
            vals=d[column].to_numpy(dtype=float)
            for idx in range(5,len(vals)):
                window=vals[idx-5:idx]
                mu, sd=fit_normal(window)
                actual=float(vals[idx])
                if sd < 1e-12:
                    # Constant historical observations imply a point-mass model.
                    low=high=mu
                    included=abs(actual-mu)<1e-9
                else:
                    a=-mu/sd
                    low=float(truncnorm.ppf(0.05, a, np.inf, loc=mu, scale=sd))
                    high=float(truncnorm.ppf(0.95, a, np.inf, loc=mu, scale=sd))
                    included=low<=actual<=high
                rows.append({
                    'Product':product, 'Duration column':column,
                    'Target observation':idx+1, 'Order ID':d.iloc[idx]['order_id'],
                    'Fitted μ (s)':mu, 'Fitted σ (s)':sd,
                    '5% bound (s)':low, '95% bound (s)':high,
                    'Observed (s)':actual,
                    'Inside 90% interval':bool(included),
                    'Deviation from μ (s)':actual-mu,
                })
    return pd.DataFrame(rows)
