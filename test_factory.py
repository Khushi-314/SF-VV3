import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from calibration import parse_data,calibrate,rolling_validation,most_recent_five
from config import MACHINES,PRODUCTS,RESOURCES,FMS_SHARES
from reliability import weibull_params
from simulation import simulate,ensemble,summarize,release_sequence,mean_wip_curve

RAW=(pathlib.Path(__file__).resolve().parents[1]/'data'/'sample_observations.csv').read_text()
DF=parse_data(RAW)
LATEST,TABLE,PARAMS=calibrate(DF)

def conf(mttf_s=200):
    return {m:{**weibull_params('MTTF + β',mttf_s,2),'repair_s':8.,'repair_cv':0.25} for m in MACHINES}

def test_data_and_calibration():
    assert len(DF)==36
    assert all(DF.groupby('product').size().reindex(PRODUCTS)==9)
    assert len(LATEST)==20
    assert len(TABLE)==48
    assert len(rolling_validation(DF))==96
    for p in PRODUCTS:
        assert set((p,m) for m in RESOURCES).issubset(PARAMS)
    assert abs(sum(FMS_SHARES.values())-1)<1e-12

def test_release_cycle_skips_exhausted_types():
    assert release_sequence({'PR1':3,'PR2':2,'PR3':0,'PR4':2}) == ['PR1','PR2','PR4','PR1','PR2','PR4','PR1']

def test_weibull_input_pairs_and_uniqueness():
    a=weibull_params('MTTF + β',600,2)
    b=weibull_params('η + β',a['eta_s'],2)
    assert abs(a['mttf_s']-b['mttf_s'])<1e-8
    assert abs(a['eta_s']-b['eta_s'])<1e-8

def test_wip_capacity_and_product_conservation():
    quantities={p:2 for p in PRODUCTS}
    for wip in (1,3,5):
        for b in (0,2):
            r=simulate(PARAMS,quantities,wip,{m:b for m in MACHINES},conf(500),seed=72,log=True)
            assert sum(r['completed'].values())==8
            assert max(v for _,v in r['wip_events'])<=wip
            assert r['wip_events'][-1][1]==0
            assert all(abs(sum(d.values())-r['makespan_s'])<1e-5 for d in r['states'].values())
            assert all(m>=0 for m in r['failures'].values())
            assert r['throughput_h']>0

def test_small_mttf_triggers_failures_and_repair_without_loss():
    r=simulate(PARAMS,{p:2 for p in PRODUCTS},3,{m:0 for m in MACHINES},conf(20),seed=3,log=True)
    assert sum(r['completed'].values())==8
    assert sum(r['failures'].values())>5
    assert sum(d['B5'] for d in r['states'].values())>0
    events=[x['Event'] for x in r['event_log']]
    assert 'ERROR_ALARM_B7' in events and 'REPAIR_COMPLETE' in events and 'RESUME' in events

def test_reproducible_monte_carlo_and_state_shares():
    q={p:1 for p in PRODUCTS}
    x=ensemble(PARAMS,q,3,{m:1 for m in MACHINES},conf(80),runs=5,seed=55)
    y=ensemble(PARAMS,q,3,{m:1 for m in MACHINES},conf(80),runs=5,seed=55)
    assert [r['makespan_s'] for r in x]==[r['makespan_s'] for r in y]
    s=summarize(x)
    for name,g in s['state_table'].groupby('Resource'):
        assert abs(g['Mean share (%)'].sum()-100)<1e-7
    assert not mean_wip_curve(x).empty

def test_sample_calibration_uses_all_five_per_product():
    for product in PRODUCTS:
        selected=LATEST[LATEST['product']==product]
        assert len(selected)==5
        assert abs(TABLE.query('Resource == "WS1" and Product == @product')['Mean (s)'].iloc[0]-selected['WS1'].mean())<1e-7

def test_final_station_is_ws8_and_uses_direct_observations():
    from config import STAGES, STAGE_OBS
    assert STAGES[-1] == 'WS8'
    assert 'WS9' not in STAGES and 'WS9' not in RESOURCES
    assert STAGE_OBS['WS8'] == 'WS8'
    for product in PRODUCTS:
        five = LATEST[LATEST['product'] == product]
        row = TABLE.query('Resource == "WS8" and Product == @product').iloc[0]
        assert abs(row['Mean (s)'] - five['WS8'].mean()) < 1e-8
        assert row['Provenance'] == 'Observed workstation'


def test_distribution_validation_is_out_of_sample_and_not_a_forecast():
    v=rolling_validation(DF)
    assert len(v)==96
    assert 'Predicted (s)' not in v.columns
    assert 'Fitted μ (s)' in v.columns
    assert 'Inside 90% interval' in v.columns
    for product in PRODUCTS:
        observations=DF[DF['product']==product].sort_values('row_sequence')
        first=v[(v['Product']==product)&(v['Duration column']=='WS1')].iloc[0]
        assert abs(first['Fitted μ (s)'] - observations['WS1'].iloc[:5].mean())<1e-9
        assert abs(first['Observed (s)']-observations['WS1'].iloc[5])<1e-9
        assert first['5% bound (s)']>=0
        assert first['95% bound (s)']>=first['5% bound (s)']


def test_arbitrary_order_id_append_is_accepted_and_used_as_latest():
    from io import StringIO
    import pandas as pd
    data = pd.read_csv(StringIO(RAW), dtype=str)
    last = data.iloc[-1].copy()
    last.iloc[0] = 'MY-NEW-ORDER-2026'
    last['Product'] = 'PR1'
    data = pd.concat([data, pd.DataFrame([last])], ignore_index=True)
    parsed = parse_data(data.to_csv(index=False))
    assert len(parsed) == 37
    assert parsed.iloc[-1]['order_id'] == 'MY-NEW-ORDER-2026'
    last_five = most_recent_five(parsed)
    assert 'MY-NEW-ORDER-2026' in set(last_five['order_id'])

def test_duplicate_arbitrary_ids_are_rejected():
    from io import StringIO
    import pandas as pd
    import pytest
    data = pd.read_csv(StringIO(RAW), dtype=str)
    data.iloc[1, 0] = data.iloc[0, 0]
    with pytest.raises(ValueError, match='Duplicate'):
        parse_data(data.to_csv(index=False))
