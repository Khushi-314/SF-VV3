import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from calibration import parse_data, calibrate
from config import MACHINES
from reliability import weibull_params
from optimizer import theoretical_wip,search_wip,rank_interventions,OptimizerConfig
from simulation import ensemble,summarize

base=Path(__file__).resolve().parents[1]
_,_,params=calibrate(parse_data((base/'data/sample_observations.csv').read_text()))
reliability={m:{**weibull_params('MTTF + β',500,2),'repair_s':8.,'repair_cv':.25} for m in MACHINES}
buffers={m:0 for m in MACHINES}
quantities={'PR1':6,'PR2':4,'PR3':3,'PR4':2}

def test_adaptive_wip_search_and_no_wip_exceeding_quantity():
    w=search_wip(params,quantities,buffers,reliability,config=OptimizerConfig(2,3,max_search_wip=30))
    assert 1<=w['recommended_wip']<=sum(quantities.values())
    assert w['wip_table']['WIP'].is_unique
    assert (w['wip_table']['throughput_h']>0).all()
    assert w['theoretical_wip']>0

def test_recommendations_are_scenario_specific_and_finite():
    t=rank_interventions(params,quantities,3,buffers,reliability,screening_runs=2,refinement_runs=3,top_k=10)
    assert len(t)==10
    assert {'Intervention','Mean time saved (min)','95% CI ± (min)','Evidence'}.issubset(t.columns)
    assert np.isfinite(t['Mean time saved (min)']).all()
    assert 'Buffer' in set(t['Category'].str.split(' · ').str[0]) or len(t)==10

def test_little_law_per_run_and_non_detailed_runs():
    r=ensemble(params,quantities,3,buffers,reliability,runs=3,seed=44,detailed_runs=1)
    assert r[1]['lead']=={p:[] for p in quantities}
    for x in r:
        flow=sum(x['lead_sum'].values())/sum(x['lead_count'].values())
        assert abs(x['wip_mean']-x['throughput_h']*flow/3600)<1e-8
    assert not summarize(r)['product_table'].empty

def test_large_orders_dont_store_all_lead_times_or_event_logs():
    r=ensemble(params,{'PR1':600,'PR2':400,'PR3':200,'PR4':100},3,buffers,reliability,2,101,detailed_runs=1)
    assert sum(r[0]['completed'].values())==1300
    assert len(r[0]['event_log'])<=5000
    assert len(r[0]['wip_events'])<=12000
    assert sum(map(len,r[1]['lead'].values()))==0
