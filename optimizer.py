"""Budget-aware finite-order simulation optimization.

Screening results are exploratory. Confidence intervals quantify Monte Carlo
precision, not uncertainty in fitted production / reliability inputs.
"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from config import PRODUCTS, MACHINES, RESOURCES, STAGES, STAGE_RESOURCES
from simulation import ensemble

@dataclass(frozen=True)
class OptimizerConfig:
    screening_runs: int = 6
    refinement_runs: int = 30
    throughput_fraction: float = 0.95
    max_search_wip: int = 96
    # Simulation-evaluation budget in total products processed across all replications.
    # The budget regulates online workload; it is not a guarantee of runtime.
    product_run_budget: int = 450000


def _ci(x):
    x=np.asarray(x,dtype=float)
    return float(1.96*x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else float('nan')


def _stats(results):
    m=np.asarray([r['makespan_s']/60 for r in results],float)
    th=np.asarray([r['throughput_h'] for r in results],float)
    flow=np.asarray([sum(r['lead_sum'].values())/max(1,sum(r['lead_count'].values()))/60 for r in results])
    return {'makespan_min':float(m.mean()),'makespan_ci_min':_ci(m),
            'throughput_h':float(th.mean()),'throughput_ci_h':_ci(th),
            'cycle_min':float(flow.mean()),'cycle_ci_min':_ci(flow),
            'mean_wip':float(np.mean([r['wip_mean'] for r in results])),
            'max_actual_wip':float(np.mean([r['wip_max'] for r in results])),
            'p95_makespan_min':float(np.percentile(m,95))}


def theoretical_wip(parameters, quantities):
    """Idealized estimate: weighted raw time × limiting stage rate."""
    total=sum(int(v) for v in quantities.values())
    if total<=0:raise ValueError('At least one product is required')
    mix={p:int(quantities.get(p,0))/total for p in PRODUCTS}
    stage_times={}
    for stage in STAGES:
        m=STAGE_RESOURCES[stage][0]
        stage_times[stage]=sum(mix[p]*parameters[(p,m)][0] for p in PRODUCTS)
    raw=sum(stage_times.values())
    cap=min(len(STAGE_RESOURCES[s])/stage_times[s] for s in STAGES if stage_times[s]>0)
    return max(1.,float(raw*cap)),min(STAGES,key=lambda s:len(STAGE_RESOURCES[s])/stage_times[s])


def _grid(estimate, upper):
    values={1,2,3,4,5,int(upper)}
    for scale in (0.5,0.75,1,1.25,1.5,2,3,4):values.add(round(estimate*scale))
    x=2
    while x<upper:
        values.add(x); x=int(1.6*x)+1
    return sorted(int(v) for v in values if 1<=v<=upper)


def _run(parameters,quantities,wip,buffers,reliability,runs,seed):
    return ensemble(parameters,quantities,wip,buffers,reliability,
                    runs=runs,seed=seed,detailed_runs=0)


def search_wip(parameters,quantities,buffers,reliability,seed=1313,config=None):
    config=config or OptimizerConfig()
    total=sum(int(v) for v in quantities.values())
    theory,bottleneck=theoretical_wip(parameters,quantities)
    # The search ceiling is a transparent safety limit, never labelled a global optimum.
    upper=min(total,config.max_search_wip,max(24,math.ceil(4*theory),2*len(RESOURCES)))
    grid=_grid(theory,upper)
    table_cache={}
    runs_used=0
    def check(w,runs):
        nonlocal runs_used
        old=table_cache.get(w)
        if old is not None and old['Runs']>=runs:return old
        result=_stats(_run(parameters,quantities,w,buffers,reliability,runs,seed))
        table_cache[w]={**result,'Runs':runs}
        runs_used+=runs # Count full runs even when recomputing, conservatively.
        return table_cache[w]
    for w in grid:check(w,config.screening_runs)
    threshold=config.throughput_fraction*max(d['throughput_h'] for d in table_cache.values())
    candidate=min(w for w,d in table_cache.items() if d['throughput_h']>=threshold)
    prior=max((x for x in grid if x<candidate),default=1)
    # Ensure integer choices at the threshold, without scanning the entire WIP range.
    for j in range(max(1,prior),candidate+1):check(j,config.screening_runs)
    threshold=config.throughput_fraction*max(d['throughput_h'] for d in table_cache.values())
    candidate=min(w for w,d in table_cache.items() if d['throughput_h']>=threshold)
    # Consistent-precision comparisons for the candidate, neighbours and high-WIP reference.
    # On large batches, a reduced number is disclosed instead of silently launching many runs.
    fine=min(config.refinement_runs,max(config.screening_runs,config.product_run_budget//max(total*12,1)))
    finalists=sorted(set([candidate,upper,*range(max(1,candidate-2),min(upper,candidate+3)+1)]))
    for w in finalists:check(w,fine)
    # Refine the actual crossing of the 95%-of-high-WIP throughput benchmark.
    ref=max(table_cache[w]['throughput_h'] for w in finalists)
    good=[w for w in finalists if table_cache[w]['throughput_h']>=config.throughput_fraction*ref]
    chosen=min(good) if good else candidate
    # Include smaller previously screened WIPs if the threshold has shifted substantially.
    candidates=[w for w in table_cache if table_cache[w]['throughput_h']>=config.throughput_fraction*ref]
    if candidates and min(candidates)<chosen:
        smaller=min(candidates)
        check(smaller,fine)
        if table_cache[smaller]['throughput_h']>=config.throughput_fraction*ref:chosen=smaller
    # Saturation evidence: upper and recommended throughputs close, with transparent precision.
    reached=upper>1 and (table_cache[upper]['throughput_h']-table_cache[chosen]['throughput_h'])<=0.05*max(table_cache[upper]['throughput_h'],1e-9)
    result=pd.DataFrame([{'WIP':w,**v} for w,v in sorted(table_cache.items())])
    return {'recommended_wip':chosen,'theoretical_wip':theory,'theoretical_bottleneck':bottleneck,
            'wip_table':result,'tested_upper':upper,'saturation_supported':bool(reached),
            'screening_runs_used':config.screening_runs,'wip_refinement_runs_used':fine,
            'note':'Finite-order throughput; saturation is estimated, not proven. WIP search is bounded.',
            'wip_experiment_runs':runs_used}


def _variants(buffers,reliability):
    for m in MACHINES:
        for frac in (.25,.50):
            c={k:v.copy() for k,v in reliability.items()}
            c[m]['mttf_s']*=1+frac;c[m]['eta_s']*=1+frac
            yield f'{m}: MTTF +{frac:.0%}',f'Reliability · {m}',c,buffers.copy()
            c={k:v.copy() for k,v in reliability.items()}
            c[m]['repair_s']*=1-frac
            yield f'{m}: repair mean -{frac:.0%}',f'Repair · {m}',c,buffers.copy()
        for increment in (1,2):
            b=buffers.copy();b[m]=b.get(m,0)+increment
            yield f'{m}: buffer +{increment}',f'Buffer · {m}',reliability,b


def rank_interventions(parameters,quantities,wip,buffers,reliability,seed=1313,
                       screening_runs=6,refinement_runs=30,top_k=10,
                       max_candidates=54,return_metadata=False):
    all_variants=list(_variants(buffers,reliability))
    # Round-robin coverage by intervention type and machine for resource-limited searches.
    if max_candidates<len(all_variants):
        # prefer smaller changes initially; do not silently claim all 54 evaluated.
        selected=[]
        for i in (0,2,4,1,3,5):
            selected.extend(all_variants[i::6])
        all_variants=selected[:max_candidates]
    baseline_screen=_run(parameters,quantities,wip,buffers,reliability,screening_runs,seed)
    bscreen=float(np.mean([r['makespan_s'] for r in baseline_screen])/60)
    ranked=[]
    for label,group,conf,buf in all_variants:
        test=_run(parameters,quantities,wip,buf,conf,screening_runs,seed)
        gain=bscreen-np.mean([r['makespan_s'] for r in test])/60
        ranked.append((float(gain),label,group,conf,buf))
    ranked.sort(key=lambda r:r[0],reverse=True)
    shortlist=ranked[:max(0,top_k-3)]
    for cat in ('Reliability','Repair','Buffer'):
        match=next((r for r in ranked if r[2].startswith(cat)),None)
        if match and match[1] not in {r[1] for r in shortlist}:shortlist.append(match)
    for r in ranked:
        if len(shortlist)>=top_k:break
        if r[1] not in {x[1] for x in shortlist}:shortlist.append(r)
    baseline=_run(parameters,quantities,wip,buffers,reliability,refinement_runs,seed)
    bm=np.asarray([r['makespan_s']/60 for r in baseline])
    records=[]
    for _,label,group,conf,buf in shortlist:
        trials=_run(parameters,quantities,wip,buf,conf,refinement_runs,seed)
        trial=np.asarray([r['makespan_s']/60 for r in trials])
        paired=bm-trial
        ci=_ci(paired)
        mean=float(paired.mean())
        status=('Supported gain' if mean>0 and mean-ci>0 else
                'Supported loss' if mean<0 and mean+ci<0 else 'Uncertain')
        records.append({'Intervention':label,'Category':group,
            'Mean time saved (min)':mean,'95% CI ± (min)':ci,
            'Improvement (%)':float(100*mean/bm.mean()),
            'P95 reduction (min)':float(np.percentile(bm,95)-np.percentile(trial,95)),
            'Evidence':status,'Refinement runs':refinement_runs})
    ranked_df=pd.DataFrame(records).sort_values('Mean time saved (min)',ascending=False).reset_index(drop=True)
    meta={'evaluated_candidates':len(all_variants),'total_candidates':54,
          'screening_runs':screening_runs,'refinement_runs':refinement_runs,
          'estimated_runs':screening_runs*(len(all_variants)+1)+refinement_runs*(len(shortlist)+1),
          'search_limited':len(all_variants)<54,
          'note':'Ranking by point estimates; different candidates may be statistically indistinguishable.'}
    return (ranked_df,meta) if return_metadata else ranked_df


def optimize(parameters,quantities,buffers,reliability,seed=1313,screening_runs=None,refinement_runs=None,
             budget=450000):
    total=sum(quantities.values())
    # Finite online workload allocation in product-run units (not guaranteed elapsed time).
    # Floor 2 screenings permits uncertainty estimates; large orders receive reduced coverage.
    max_runs=max(2,budget//max(total,1))
    screen=min(6,max(2,max_runs//80)) if screening_runs is None else screening_runs
    fine=min(30,max(3,max_runs//25)) if refinement_runs is None else refinement_runs
    if fine<screen:fine=screen
    conf=OptimizerConfig(screening_runs=screen,refinement_runs=fine,product_run_budget=budget)
    w=search_wip(parameters,quantities,buffers,reliability,seed,conf)
    # Limit number of intervention alternatives on huge batches; disclose coverage.
    covered=54 if max_runs>=100 else (18 if max_runs>=30 else 9)
    result,meta=rank_interventions(parameters,quantities,w['recommended_wip'],buffers,reliability,
            seed,screen,fine,max_candidates=covered,return_metadata=True)
    w['recommendations']=result
    w['search_metadata']=meta
    w['estimated_total_optimizer_runs']=w['wip_experiment_runs']+meta['estimated_runs']
    w['budget_note']='Workload budget is heuristic, not a wall-time guarantee; large orders may remain slow.'
    return w
