import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from calibration import calibrate, parse_data
from config import MACHINES
from reliability import weibull_params
from optimizer import OptimizerConfig,search_wip,rank_interventions,optimize
from simulation import simulate

P=Path(__file__).resolve().parents[1]
_,_,params=calibrate(parse_data((P/'data/sample_observations.csv').read_text()))
rs={m:{**weibull_params('MTTF + β',500,2),'repair_s':8.,'repair_cv':0.25} for m in MACHINES}
buffs={m:0 for m in MACHINES}
q={'PR1':6,'PR2':4,'PR3':3,'PR4':2}

def test_peak_actual_wip_is_bounded_and_distinct_from_limit():
    for w in (1,3,10):
        v=simulate(params,q,w,buffs,rs,seed=119,trace=False,detailed=False)
        assert 0<v['wip_max']<=w
        assert v['wip_mean']<=v['wip_max']+1e-8

def test_wip_search_reports_precision_and_actual_wip():
    o=search_wip(params,q,buffs,rs,seed=75,config=OptimizerConfig(2,3,max_search_wip=20))
    assert {'WIP','mean_wip','max_actual_wip','Runs','throughput_ci_h'}.issubset(o['wip_table'].columns)
    assert o['wip_table']['WIP'].max()<=sum(q.values())
    assert o['wip_table']['max_actual_wip'].le(o['wip_table']['WIP']).all()
    assert o['wip_experiment_runs']>=len(o['wip_table'])*2

def test_limited_intervention_catalogue_is_explicit():
    df,meta=rank_interventions(params,q,3,buffs,rs,seed=75,screening_runs=2,
        refinement_runs=3,max_candidates=9,return_metadata=True)
    assert meta['evaluated_candidates']==9
    assert meta['search_limited']
    assert meta['total_candidates']==54
    assert len(df)<=9
    assert df['Refinement runs'].eq(3).all()

def test_optimizer_does_not_need_user_optional_wip_inputs():
    result=optimize(params,q,buffs,rs,seed=55,screening_runs=2,refinement_runs=3)
    assert result['recommended_wip']>0
    assert result['search_metadata']['evaluated_candidates']>=9
    assert result['estimated_total_optimizer_runs']>0
