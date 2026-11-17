"""Machine-level discrete-event factory simulation.

One engine for a single run or a Monte Carlo ensemble. All times in seconds.
Events are process finishes, active-use Weibull breakdowns and repairs.
Every resource has capacity one; queue capacity excludes the processing slot.
Strict alternation (not availability-skipping) at paired workstations.
"""
from collections import deque
from dataclasses import dataclass
import heapq
import itertools
import math
import numpy as np
import pandas as pd
from config import STAGES,STAGE_RESOURCES,PRODUCTS,RESOURCES,MACHINES,FMS_SHARES
from reliability import sample_lifetime,sample_repair

@dataclass
class Job:
    id: int
    product: str
    stage: int = 0
    released: float = 0.0
    remaining: float|None = None
    next_machine: str|None = None
    fms_total: float|None = None


def positive_normal(rng,mu,sd):
    """Normal conditioned to positive times, rather than negative or zero durations."""
    if mu<=0 or sd<0: raise ValueError('Invalid processing distribution')
    if sd==0:return float(mu)
    for _ in range(1000):
        v=float(rng.normal(mu,sd))
        if v>0: return v
    raise RuntimeError('Positive-normal rejection sampling failed; inspect data')

def release_sequence(quantities):
    left={p:int(quantities.get(p,0)) for p in PRODUCTS}
    if any(n<0 for n in left.values()): raise ValueError('Order quantities must be nonnegative')
    out=[]
    while any(left.values()):
        for p in PRODUCTS:
            if left[p]>0:out.append(p);left[p]-=1
    return out


def simulate(parameters,quantities,wip_limit=1,buffers=None,reliability=None,seed=123,horizon_s=5e7,log=False,
             detailed=True, max_log=5000, max_lead_samples=3000, max_wip_events=12000, trace=True):
    if int(wip_limit)<1:raise ValueError('WIP must be >=1')
    order=release_sequence(quantities)
    rng=np.random.default_rng(seed)
    buffs={m:0 for m in MACHINES}
    if buffers:buffs.update({m:int(v) for m,v in buffers.items()})
    if any(x<0 for x in buffs.values()):raise ValueError('Buffer capacity must be nonnegative')
    for p in set(order):
        for m in RESOURCES:
            if (p,m) not in parameters:raise ValueError(f'Missing duration for {p}/{m}')
    reliability=reliability or {}
    res={}
    for name in RESOURCES:
        res[name]=dict(state='B1',job=None,queue=deque(),hazard=None,
             stamp=0.0,change_time=0.0,spent={s:0.0 for s in ('B1','B3','B5','B6')},
             fails=0,cycles=0,token=0,working_since=None,queue_area=0.0,
             max_queue=0,queue_stamp=0.0,waiting_total=0.0)
    rr={s:0 for s in STAGES if len(STAGE_RESOURCES[s])==2}
    events=[]; counter=itertools.count(); clk=0.0
    next_admit=0;wip=0;wip_peak=0;wip_integral=0.0;wip_events=[(0.0,0)]
    completed={p:0 for p in PRODUCTS};lead={p:[] for p in PRODUCTS};detail=[]
    event_log=[]
    log_truncated=False
    lead_sum={p:0.0 for p in PRODUCTS}; lead_count={p:0 for p in PRODUCTS}
    sampling_rng=np.random.default_rng(int(seed)^0x5F3759DF)
    wip_stride=1

    def keep_wip_event():
        nonlocal wip_stride,wip_events
        if not trace:return
        if len(wip_events)>=max_wip_events:
            wip_events=wip_events[::2]
            wip_stride*=2
        if next_admit % wip_stride==0 or not wip:
            wip_events.append((clk,wip))

    def record(message,resource='',job=None):
        nonlocal log_truncated
        if log and len(event_log)<max_log:
            event_log.append({'Time (s)':round(clk,5),'Resource':resource,
              'Product':job.product if job else '', 'Unit ID':job.id if job else '', 'Event':message})
        elif log:log_truncated=True
    def change(name,state):
        r=res[name]
        if r['state']!=state:
            r['spent'][r['state']]+=clk-r['change_time']
            r['change_time']=clk;r['state']=state
    def queue_update(r):
        r['queue_area']+=len(r['queue'])*(clk-r['queue_stamp'])
        r['queue_stamp']=clk
    def schedule(delay,etype,name,token):
        if delay<0:raise RuntimeError('Negative event delay')
        heapq.heappush(events,(clk+delay,next(counter),etype,name,token))
    def assign(job):
        st=STAGES[job.stage]; options=STAGE_RESOURCES[st]
        if len(options)==1:return options[0]
        selected=options[rr[st]%len(options)]
        rr[st]+=1
        return selected
    def sample_duration(job,name):
        stage=STAGES[job.stage]
        if stage in ('WS5','WS6','WS7'):
            if job.fms_total is None:
                mu,sd=parameters[(job.product,'MC5')]
                job.fms_total=positive_normal(rng,mu/FMS_SHARES['WS5'],sd/FMS_SHARES['WS5'])
            return job.fms_total*FMS_SHARES[stage]
        mu,sd=parameters[(job.product,name)]
        return positive_normal(rng,mu,sd)
    def start(name,job,continuing=False):
        r=res[name]
        if r['job'] is not None and r['job'] is not job:raise AssertionError('Resource double-booked')
        r['job']=job
        if not continuing:
            if job.remaining is not None:raise AssertionError('Nonzero remaining work on new stage')
            job.remaining=sample_duration(job,name)
        change(name,'B3')
        r['stamp']=clk;r['token']+=1;token=r['token']
        if name in MACHINES:
            conf=reliability[name]
            if r['hazard'] is None:
                r['hazard']=sample_lifetime(rng,conf['eta_s'],conf['beta'])
            if r['hazard']<job.remaining:
                schedule(max(0,r['hazard']),'failure',name,token)
            else:schedule(job.remaining,'finished',name,token)
        else:schedule(job.remaining,'finished',name,token)
        record('RESUME' if continuing else 'START_PROCESSING',name,job)

    def accept(name,job):
        r=res[name]
        if r['job'] is None and r['state']=='B1' and not r['queue']:
            start(name,job);return True
        if len(r['queue'])<buffs.get(name,0):
            queue_update(r);r['queue'].append(job)
            r['max_queue']=max(r['max_queue'],len(r['queue']))
            record('ENTER_BUFFER',name,job)
            return True
        return False

    def unblock():
        # Every blocked machine retains its completed job; retry when a downstream slot opens.
        changed=True
        while changed:
            changed=False
            for origin in RESOURCES:
                r=res[origin];job=r['job']
                if r['state']!='B6' or job is None:continue
                dest=job.next_machine
                if accept(dest,job):
                    record('TRANSFER',origin,job)
                    r['job']=None;job.next_machine=None
                    change(origin,'B1')
                    launch_waiting(origin)
                    changed=True

    def launch_waiting(name):
        r=res[name]
        if r['job'] is None and r['state']=='B1' and r['queue']:
            queue_update(r);job=r['queue'].popleft()
            start(name,job)

    def attempt_admit():
        nonlocal next_admit,wip,wip_peak,wip_events
        entrance=res['WS1']
        while next_admit<len(order) and wip<int(wip_limit):
            if entrance['job'] is not None or entrance['queue']:break
            job=Job(next_admit+1,order[next_admit],released=clk)
            wip+=1;wip_peak=max(wip_peak,wip);next_admit+=1;keep_wip_event()
            record('RELEASE','WS1',job)
            start('WS1',job)

    def finish_or_transfer(name):
        nonlocal wip
        r=res[name];job=r['job']
        if job.stage==len(STAGES)-1:
            completed[job.product]+=1
            duration=clk-job.released
            lead_count[job.product]+=1;lead_sum[job.product]+=duration
            if detailed:
                sample=lead[job.product];k=lead_count[job.product]
                if len(sample)<max_lead_samples:sample.append(duration)
                else:
                    idx=int(sampling_rng.integers(k))
                    if idx<max_lead_samples:sample[idx]=duration
            record('COMPLETE',name,job)
            wip-=1;keep_wip_event()
            r['job']=None;change(name,'B1');launch_waiting(name)
            unblock();attempt_admit()
            return
        job.stage+=1;job.remaining=None
        job.next_machine=assign(job)
        record('STAGE_COMPLETE',name,job)
        if accept(job.next_machine,job):
            r['job']=None;job.next_machine=None
            change(name,'B1');launch_waiting(name)
            unblock();attempt_admit()
        else:
            change(name,'B6');record('WAIT_FOR_MACHINE',name,job)

    attempt_admit()
    if not order:
        return {'makespan_s':0.,'throughput_h':0.,'completed':completed,
                'failures':{m:0 for m in MACHINES},'states':{m:{'B1':0.,'B3':0.,'B5':0.,'B6':0.} for m in RESOURCES},
                'lead':lead,'wip_mean':0.,'wip_max':0,'wip_events':wip_events,'queue_mean':{},'queue_max':{},
                'event_log':event_log,'deadlocked':False,'requested':0,
                'lead_sum':lead_sum,'lead_count':lead_count,'log_truncated':False}
    last_clock=0.;steps=0
    while events:
        t,_,typ,name,token=heapq.heappop(events)
        if t>horizon_s:raise RuntimeError('Simulation exceeded horizon; lower batch/MTTF or inspect blocking')
        wip_integral+=wip*(t-clk);clk=t
        r=res[name]
        if token!=r['token']:continue
        job=r['job']
        if typ=='repair':
            if r['state']!='B5' or job is None:raise AssertionError('Invalid repair completion')
            conf=reliability[name]
            r['hazard']=sample_lifetime(rng,conf['eta_s'],conf['beta'])
            record('REPAIR_COMPLETE',name,job)
            start(name,job,continuing=True)
        elif typ in ('finished','failure'):
            if r['state']!='B3' or job is None:raise AssertionError('Invalid process event')
            work=clk-r['stamp'];job.remaining=max(0,job.remaining-work)
            if name in MACHINES:r['hazard']=max(0,r['hazard']-work)
            if typ=='failure':
                r['fails']+=1
                record('ERROR_ALARM_B7',name,job)
                change(name,'B5')
                record('MAINTENANCE_START_B5',name,job)
                conf=reliability[name]
                repair=sample_repair(rng,conf['repair_s'],conf['repair_cv'])
                schedule(repair,'repair',name,r['token'])
            else:
                r['cycles']+=1
                job.remaining=0.
                finish_or_transfer(name)
        else:raise AssertionError('Unknown event')
        steps+=1
        if steps>4_000_000:raise RuntimeError('Run event limit exceeded')
        if sum(completed.values())==len(order):break
    deadlocked=(sum(completed.values())!=len(order))
    for name,r in res.items():
        r['spent'][r['state']]+=clk-r['change_time']
        queue_update(r)
        if abs(sum(r['spent'].values())-clk)>1e-6*max(clk,1):
            raise AssertionError(f'State accounting error {name}')
    if wip<0 or wip>int(wip_limit):raise AssertionError('WIP conservation failure')
    if deadlocked:raise RuntimeError('Deadlock: unfinished products with no scheduled events')
    makespan=max(clk,1e-10)
    return dict(makespan_s=clk,throughput_h=len(order)*3600/makespan,
        completed=completed,failures={m:res[m]['fails'] for m in MACHINES},
        states={m:res[m]['spent'] for m in RESOURCES},
        lead=lead,wip_mean=wip_integral/makespan,wip_max=wip_peak,wip_events=wip_events,
        queue_mean={m:res[m]['queue_area']/makespan for m in RESOURCES},
        queue_max={m:res[m]['max_queue'] for m in RESOURCES},
        event_log=event_log,deadlocked=False,requested=len(order),
        lead_sum=lead_sum,lead_count=lead_count,log_truncated=log_truncated)


def ensemble(parameters,quantities,wip_limit,buffers,reliability,runs=100,seed=1313,
             detailed_runs=12):
    if not 1<=int(runs)<=10000:raise ValueError('Runs must be between 1 and 10,000')
    results=[]
    for k in range(int(runs)):
        # Record detailed log for one known replication only (diagnostic, never averaged).
        results.append(simulate(parameters,quantities,wip_limit,buffers,reliability,
                                seed=int(seed)+k*104729,log=(k==0),detailed=(k<int(detailed_runs)),
                                trace=(int(detailed_runs)>0),
                                max_wip_events=(12000 if k<int(detailed_runs) else 2000)))
    return results

def summarize(results):
    if not results:return {}
    n=len(results)
    makes=np.array([x['makespan_s']/60 for x in results]);th=np.array([x['throughput_h'] for x in results])
    states=[];failures=[];product_rows=[]
    for name in RESOURCES:
        for state in ('B1','B3','B5','B6'):
            times=np.array([x['states'][name][state] for x in results])
            pct=np.array([100*x['states'][name][state]/x['makespan_s'] if x['makespan_s'] else 0 for x in results])
            ci=float(1.96*pct.std(ddof=1)/np.sqrt(n)) if n>1 else 0.
            states.append({'Resource':name,'State':state,'Mean time (min)':float(times.mean()/60),
                           'Mean share (%)':float(pct.mean()), '95% CI half-width (%)':ci})
        if name in MACHINES:
            failures.append({'Machine':name,'Mean failures (count/run)':float(np.mean([x['failures'][name] for x in results])),
                             'Mean maintenance (min/run)':float(np.mean([x['states'][name]['B5']/60 for x in results]))})
    for product in PRODUCTS:
        l=[v for x in results for v in x['lead'][product]]
        product_rows.append({'Product':product,'Mean completed (products/run)':float(np.mean([x['completed'][product] for x in results])),
            'Mean lead (s)':(sum(x['lead_sum'][product] for x in results)/
              max(1,sum(x['lead_count'][product] for x in results))),
            'P95 lead (s)':float(np.percentile(l,95)) if l else 0.})
    return {'makespans':makes,'throughput':th,'mean_makespan_min':float(makes.mean()),
        'p95_makespan_min':float(np.percentile(makes,95)), 'mean_throughput_h':float(th.mean()),
        'mean_wip':float(np.mean([x['wip_mean'] for x in results])),
        'mean_max_wip':float(np.mean([x['wip_max'] for x in results])),
        'mean_failures':float(np.mean([sum(x['failures'].values()) for x in results])),
        'state_table':pd.DataFrame(states),'failures_table':pd.DataFrame(failures),
        'product_table':pd.DataFrame(product_rows)}

def mean_wip_curve(results,steps=100):
    results=[x for x in results if len(x['wip_events'])>1]
    if not results:return pd.DataFrame({'Time (min)':[0.], 'Mean WIP (products)':[0.]})
    max_time=max(x['makespan_s'] for x in results)
    if max_time==0:return pd.DataFrame({'Time (min)':[0.], 'Mean WIP (products)':[0.]})
    times=np.linspace(0,max_time,steps)
    vectors=[]
    for run in results:
        events=run['wip_events']
        t=np.array([e[0] for e in events]);v=np.array([e[1] for e in events])
        loc=np.searchsorted(t,times,side='right')-1
        arr=v[np.maximum(loc,0)]
        arr[times>=run['makespan_s']]=0
        vectors.append(arr)
    return pd.DataFrame({'Time (min)':times/60,'Mean WIP (products)':np.mean(vectors,axis=0)})
