"""Factory Decision Lab — Monte Carlo factory decision support."""
from datetime import datetime, timezone
import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config import (SHEET_URL,PRODUCTS,MACHINES,RESOURCES,STAGE_RESOURCES,
                    B_STATES,FMS_SHARES,DEFAULT_BETA,DEFAULT_MTTF_S,DEFAULT_REPAIR_S,DEFAULT_REPAIR_CV)
from calibration import fetch_live,parse_data,calibrate,rolling_validation,most_recent_five
from reliability import weibull_params
from simulation import ensemble,summarize,mean_wip_curve
from optimizer import optimize

st.set_page_config(page_title='Factory Decision Lab',page_icon='🏭',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{padding-top:1.55rem;max-width:1450px}
[data-testid="stMetric"]{border:1px solid #e2e8f0;border-radius:10px;padding:13px}
[data-testid="stSidebar"] .stHeading{margin-top:0.6rem}
</style>''',unsafe_allow_html=True)
st.title('Factory Decision Lab')
st.caption('Live product-wise calibration · Individual machine reliability · Discrete-event Monte Carlo simulation')

if 'previous' not in st.session_state:st.session_state.previous=None
if 'current' not in st.session_state:st.session_state.current=None

with st.sidebar:
    st.header('Simulation settings')
    quantities={p:st.number_input(f'{p} quantity (products)',min_value=0,max_value=200000,value=5,step=1,key=f'q_{p}') for p in PRODUCTS}
    st.markdown('**Optional comparison WIP limits** (0 = unused; up to three)')
    requested_wips=[st.number_input(f'Additional WIP {i+1}',min_value=0,max_value=200000,value=0,step=1,key=f'extra_wip_{i}') for i in range(3)]
    runs=st.number_input('Monte Carlo replications (runs)',min_value=1,max_value=1000,value=100,step=10)
    with st.expander('Buffers by machine (products)',expanded=False):
        buffs={m:st.number_input(m,min_value=0,max_value=200,value=0,step=1,key=f'buf_{m}') for m in MACHINES}

with st.expander('Machine reliability and repair settings',expanded=True):
    st.subheader('Machine reliability and repair parameters')
    mode=st.radio('Enter two parameters',options=['MTTF + β','η + β'],horizontal=True)
    st.caption('Times entered in minutes; β has no units. Third parameter calculated automatically.')
    first_label='MTTF (min)' if mode=='MTTF + β' else 'η (min)'
    default_first=90.0 if mode=='MTTF + β' else 100.0
    defaults=pd.DataFrame({'Machine':list(MACHINES),first_label:[default_first]*len(MACHINES),
                           'β':[DEFAULT_BETA]*len(MACHINES),
                           'Repair mean (min)':[DEFAULT_REPAIR_S/60]*len(MACHINES),
                           'Repair CV':[DEFAULT_REPAIR_CV]*len(MACHINES)})
    inputs=st.data_editor(defaults,hide_index=True,use_container_width=True,
      disabled=['Machine'],key=f'weibull_editor_{mode}',num_rows='fixed',
      column_config={first_label:st.column_config.NumberColumn(first_label,min_value=0.001,format='%.2f'),
        'β':st.column_config.NumberColumn('β',min_value=0.05,format='%.3f'),
        'Repair mean (min)':st.column_config.NumberColumn('Repair mean (min)',min_value=0.,format='%.2f'),
        'Repair CV':st.column_config.NumberColumn('Repair CV',min_value=0.,format='%.3f')})
    st.caption('Reliability applies only to MC1–MC9; manual workstations do not fail.')
    seed=int(st.number_input('Random seed (advanced)',min_value=0,max_value=2_000_000_000,value=1313,step=1))


go_run=st.button('Simulate & Recommend',type='primary')

try:
    reliability={}
    for _,r in inputs.iterrows():
        calc=weibull_params(mode,float(r[first_label])*60,float(r['β']))
        reliability[r['Machine']]={**calc,'repair_s':float(r['Repair mean (min)'])*60,
                                   'repair_cv':float(r['Repair CV'])}
except (ValueError,OverflowError) as ex:
    st.error(f'Invalid reliability settings: {ex}');st.stop()

if go_run:
    if not any(quantities.values()):
        st.warning('Enter at least one product to simulate.')
    else:
        try:
            with st.spinner('Fetching fresh production data from Google Sheets…'):
                live,checksum=fetch_live()
                latest,params,model=calibrate(live)
                validation=rolling_validation(live)
            with st.spinner(f'Running {runs:,} independent factory replications…'):
                if len(set(v for v in requested_wips if v>0)) != sum(v>0 for v in requested_wips):
                    raise ValueError('Additional WIP values must be distinct.')
                total_units=sum(quantities.values())
                if any(v>total_units for v in requested_wips):
                    raise ValueError('Additional WIP cannot exceed total requested products.')
                optimization=optimize(model,quantities,buffs,reliability,seed=seed)
                # The optimizer reports its evaluated search coverage and Monte Carlo precision.
                # Large-order runs remain exploratory on shared Streamlit infrastructure.
                recommended=optimization['recommended_wip']
                view_wips=list(dict.fromkeys([recommended]+[v for v in requested_wips if v>0]))
                result_by_wip={}
                for selected_wip in view_wips:
                    rr=ensemble(model,quantities,selected_wip,buffs,reliability,int(runs),seed)
                    result_by_wip[selected_wip]={'results':rr,'summary':summarize(rr)}
                results=result_by_wip[recommended]['results']
                summary=result_by_wip[recommended]['summary']
            snapshot={'df':live,'hash':checksum,'latest':latest,'params':params,
                      'validation':validation,'results':results,'summary':summary,
                      'runs':runs,'time':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
                      'reliability':reliability,'quantities':quantities.copy(),
                      'wip_limit':recommended,'buffers':buffs.copy(),'seed':seed,
                      'optimization':optimization,'result_by_wip':result_by_wip}
            st.session_state.previous=st.session_state.current
            st.session_state.current=snapshot
            st.success(f'Fetched {len(live)} valid observations; {runs} simulation runs completed. Data version {checksum}.')
        except Exception as ex:
            st.error(f'Fresh-data simulation failed: {type(ex).__name__}: {ex}')
            st.warning('No stale-data simulation was substituted. Check the published CSV link, data, and settings.')

snap=st.session_state.current
if snap is None:
    st.info('Configure orders and reliability in the sidebar, then click **Run simulation**. Every run first fetches the latest published Google Sheet.')
    st.caption('Live data source: '+SHEET_URL)
    with st.expander('Modelling assumptions',expanded=False):
        st.markdown('''- Processing: positive-truncated Normal for each product/resource, fitted from the latest five valid orders per product.
- WS5–WS7: observed aggregate allocated using indicative historical stage shares; allocation is a model assumption.
- Parallel resources: strict alternating assignments; jobs wait for their assigned machine.
- Global WIP 1 means a single admitted product in the factory. Machine queue capacities default to zero.
- Failures: Weibull operating-age; perfect lognormal repair; preempt-resume, no product loss.
- WS8 is the final modelled workstation; its duration is taken from the sheet's WS8 column.
- B6 Waiting for machine used for a resource holding a completed job unable to transfer downstream.
- B7 Error/Alarm is a timestamped failure event, not fabricated elapsed duration.''')
    st.stop()

st.caption(f"Source: **published live Google Sheet** · Last successful fetch **{snap['time']}** · "
           f"{len(snap['df'])} valid orders · Last chronological order **{snap['df'].iloc[-1]['order_id']}** · "
           f"Data version `{snap['hash']}` · {snap['runs']} replications")

st.subheader('WIP results to inspect')
selected_wip=st.selectbox('Selected WIP (existing graphs below use this value)',
                          list(snap['result_by_wip']),
                          format_func=lambda x: f'WIP {x}' + (' — recommended' if x==snap['wip_limit'] else ' — requested'))
selected_data=snap['result_by_wip'][selected_wip]
S=selected_data['summary']
R=selected_data['results']

tab_sim,tab_params,tab_validation,tab_opt=st.tabs(['Simulation results','Processing-time parameters','Live data & validation','Optimization & recommendations'])

with tab_sim:
    n=snap['runs']
    kpi=st.columns(5)
    kpi[0].metric('Mean makespan',f"{S['mean_makespan_min']:.1f} min",f"P95: {S['p95_makespan_min']:.1f} min",delta_color='off')
    kpi[1].metric('Mean throughput',f"{S['mean_throughput_h']:.2f} units/h")
    kpi[2].metric('Completed / run',f"{np.mean([sum(r['completed'].values()) for r in R]):.1f} units")
    kpi[3].metric('Mean failures',f"{S['mean_failures']:.2f} / run")
    kpi[4].metric('Mean WIP',f"{S['mean_wip']:.2f} units",f"Limit: {selected_wip} units",delta_color='off')
    if n==1:st.warning('One run is a trajectory, not a stable estimate of expected performance. Increase replications for decision-making.')
    st.subheader('Production performance')
    left,right=st.columns(2)
    with left:
        fig=go.Figure(go.Histogram(x=S['makespans'],nbinsx=min(35,max(1,int(np.sqrt(n)*2))),marker_color='#4468aa'))
        fig.add_vline(x=S['mean_makespan_min'],line_dash='dash',annotation_text='Mean')
        fig.add_vline(x=S['p95_makespan_min'],line_dash='dot',annotation_text='P95')
        fig.update_layout(title='Makespan distribution',xaxis_title='Makespan (minutes)',yaxis_title='Monte Carlo runs (count)',height=350,margin=dict(l=30,r=20,t=45,b=30))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        f=S['failures_table']
        fig=px.bar(f,x='Machine',y='Mean failures (count/run)',title='Expected breakdowns by machine',text_auto='.2f')
        fig.update_layout(height=350,margin=dict(l=30,r=20,t=45,b=30))
        st.plotly_chart(fig,use_container_width=True)
    st.subheader('Resource operations — averages across all runs')
    states=S['state_table']
    busy=states.loc[states['State']=='B3'].copy()
    busy['label']=[f"{x:.1f}% | {y:.1f} min" for x,y in zip(busy['Mean share (%)'],busy['Mean time (min)'])]
    fig=px.bar(busy,x='Resource',y='Mean share (%)',text='label',
                custom_data=['Mean time (min)'],title='Mean processing utilization',
                error_y='95% CI half-width (%)')
    fig.update_traces(textposition='outside',hovertemplate='%{x}<br>Processing: %{y:.2f}%<br>Mean time: %{customdata[0]:.2f} min/run<extra></extra>')
    fig.update_yaxes(title_text='Processing time / run (%)',range=[0,max(115,busy['Mean share (%)'].max()+18)])
    fig.update_layout(height=385,margin=dict(l=30,r=30,t=45,b=30))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('**Machine state-time composition** · horizontal position represents the mean fraction of a run. '
                'Each segment also displays its mean **absolute duration (minutes/run)**.')
    colors={'B1':'#9ca3af','B3':'#16856a','B5':'#c85b66','B6':'#d9a541'}
    fig=go.Figure()
    for state in ('B3','B1','B5','B6'):
        sub=states[states['State']==state].set_index('Resource').reindex(RESOURCES)
        pct=sub['Mean share (%)'].to_numpy();dur=sub['Mean time (min)'].to_numpy()
        labels=[f'{p:.0f}% | {d:.1f}m' if p>=10 else '' for p,d in zip(pct,dur)]
        fig.add_trace(go.Bar(name=f'{state} — {B_STATES[state]}',y=RESOURCES,x=pct,
            orientation='h',marker_color=colors[state],text=labels,textposition='inside',
            customdata=dur,hovertemplate='%{y}<br>%{fullData.name}<br>Mean fraction: %{x:.2f}%<br>Mean absolute: %{customdata:.2f} min/run<extra></extra>'))
    fig.update_layout(barmode='stack',height=515,xaxis_title='Mean resource state share of run (%)',yaxis_title='Resource',
      xaxis=dict(range=[0,100]),legend=dict(orientation='h',y=1.12),margin=dict(l=30,r=20,t=80,b=40))
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Because each run has a different makespan, one universal minutes axis cannot exactly represent mean percentages. '
               'Mean absolute durations are labelled in segments, tooltips, and the detailed table.')
    st.subheader('Work in process')
    wcurve=mean_wip_curve(R)
    wfig=px.line(wcurve,x='Time (min)',y='Mean WIP (products)',title='Mean WIP by elapsed time')
    wfig.update_layout(height=330,margin=dict(l=30,r=20,t=45,b=30))
    st.plotly_chart(wfig,use_container_width=True)
    st.caption('After a replication finishes, its WIP is treated as zero for the ensemble time-axis average. ' 
               'Large-order WIP traces are downsampled for display; mean WIP is calculated from the exact area integral.')
    with st.expander('Resource statistics: mean time, percentages, failures, queues'):
        pivot_pct=states.pivot(index='Resource',columns='State',values='Mean share (%)')
        pivot_min=states.pivot(index='Resource',columns='State',values='Mean time (min)')
        detail=[]
        for m in RESOURCES:
            row={'Resource':m}
            for code in ('B1','B3','B5','B6'):
                row[f'{B_STATES[code]} (%)']=pivot_pct.loc[m,code]
                row[f'{B_STATES[code]} (min/run)']=pivot_min.loc[m,code]
            row['Failures (count/run)']=float(np.mean([r['failures'].get(m,0) for r in R]))
            row['Avg queue length (products)']=float(np.mean([r['queue_mean'].get(m,0) for r in R]))
            row['Max queue length (products)']=max(r['queue_max'].get(m,0) for r in R)
            row['Buffer cap (products)']=snap['buffers'].get(m,0)
            detail.append(row)
        st.dataframe(pd.DataFrame(detail).round(3),hide_index=True,use_container_width=True)
        st.caption('Codes B2, B4 and B8 are defined in the source workbook but have no fabricated intervals in this simulator. B7 is logged as an alarm event.')
    with st.expander('Product performance (all replications)'):
        st.dataframe(S['product_table'].round(2),hide_index=True,use_container_width=True)
    with st.expander('Diagnostic event log — replication 1 only (NOT an ensemble statistic)'):
        st.caption('A chronological trace to audit engine behavior. This is not a mean trajectory.')
        st.dataframe(pd.DataFrame(R[0]['event_log']),hide_index=True,use_container_width=True)
        if R[0].get('log_truncated'):st.caption('Diagnostic event log capped at 5,000 rows to protect memory for large orders.')
    with st.expander('Model assumptions and limitations'):
        st.markdown(f'''**Processing:** positive-truncated Normal; five most recent valid observations per product; product-specific μ and σ.  
**Paired workstations:** strict independent alternation, starting at the lower-number machine; unavailable assigned machines are waited for.  
**WS5–WS7 shares (assumed):** WS5 {FMS_SHARES['WS5']:.1%}, WS6 {FMS_SHARES['WS6']:.1%}, WS7 {FMS_SHARES['WS7']:.1%}. The three durations from one combined observation are correlated.  
**Final workstation:** WS8, calibrated directly from the live sheet's WS8 column.  
**Reliability:** usage-only aging, Weibull lifetime; failure triggers B7 alarm event, B5 repair duration; perfect repair, preempt-resume.  
**Source state labels:** B1 Idle, B3 Processing, B5 Under maintenance, B6 Waiting for machine. B2 Setup, B4 Off, B8 Not applicable are not simulated as durations.  
**WIP:** total admitted but unfinished units. **Buffer:** waiting places outside a machine's one active processing slot.  
**Validation boundary:** rolling held-out duration checks assess the fitted processing-time distributions, not machine reliability or measured whole-factory makespan. This is not a standalone forecasting model.''')

with tab_params:
    st.subheader('Calculated Normal processing-time parameters')
    st.caption('These are the exact μ and σ inputs used for this simulation, using the current live Sheet snapshot. '
               'Each product contributes its latest five valid chronological observations.')
    params=snap['params'].copy()
    selected=st.multiselect('Filter products',PRODUCTS,default=list(PRODUCTS))
    st.dataframe(params.loc[params['Product'].isin(selected)].round(4),hide_index=True,use_container_width=True,height=540)
    st.download_button('Download parameter table (CSV)',params.to_csv(index=False).encode(),
        file_name=f'processing_parameters_{snap["hash"]}.csv',mime='text/csv')
    st.markdown('**Latest five observations selected for each product**')
    view=snap['latest'][['row_sequence','order_id','product','WS1','WS2','WS3','WS4','WS5-7','WS8']].copy()
    st.dataframe(view,hide_index=True,use_container_width=True)
    st.markdown('**Current machine reliability parameters**')
    rtab=pd.DataFrame([{'Machine':m,'MTTF (min)':v['mttf_s']/60,'η (min)':v['eta_s']/60,
                          'β':v['beta'],'Repair mean (min)':v['repair_s']/60,
                          'Repair CV':v['repair_cv']} for m,v in snap['reliability'].items()])
    st.dataframe(rtab.round(3),hide_index=True,use_container_width=True)
    prev=st.session_state.previous
    if prev is not None:
        st.markdown('**Change from previous simulation**')
        curr=params.set_index(['Resource','Product'])[['Mean (s)','SD (s)']]
        old=prev['params'].set_index(['Resource','Product'])[['Mean (s)','SD (s)']]
        comparison=curr.join(old,lsuffix=' now',rsuffix=' before')
        comparison['Δ mean (s)']=comparison['Mean (s) now']-comparison['Mean (s) before']
        st.dataframe(comparison.reset_index().round(3),hide_index=True,use_container_width=True)
        st.metric('Change in mean makespan',f"{snap['summary']['mean_makespan_min']-prev['summary']['mean_makespan_min']:+.2f} min")
        st.caption('Changes are causal comparisons only if machine settings, requested products, and random seeds were held fixed.')

with tab_validation:
    st.subheader('Live data and product-wise rolling validation')
    st.markdown(f"**Source:** [Published Google Sheet]({SHEET_URL})")
    st.caption(f"Fetched {snap['time']} · Version {snap['hash']} · {len(snap['df'])} valid historical records")
    counts=snap['df'].groupby('product').size().reindex(PRODUCTS)
    st.write('Historical observations by product:',', '.join(f'{p}: {int(counts[p])}' for p in PRODUCTS))
    product=st.selectbox('Product to validate',PRODUCTS)
    metric=st.selectbox('Observed duration column',['WS1','WS2','WS3','WS4','WS5-7','WS8'])
    v=snap['validation']
    subset=v[(v['Product']==product)&(v['Duration column']==metric)].copy()
    coverage=float(subset['Inside 90% interval'].mean()*100)
    mean_abs_dev=float(subset['Deviation from μ (s)'].abs().mean())
    cc=st.columns(3)
    cc[0].metric('Held-out observations',len(subset))
    cc[1].metric('Inside 90% model interval',f'{coverage:.1f}%')
    cc[2].metric('Mean absolute deviation from μ',f'{mean_abs_dev:.2f} s')
    fig=go.Figure()
    x=subset['Target observation'].tolist()
    fig.add_trace(go.Scatter(
        x=x,y=subset['95% bound (s)'],mode='lines',line=dict(width=0),
        showlegend=False,hovertemplate='Upper 95% bound: %{y:.2f} s<extra></extra>'))
    fig.add_trace(go.Scatter(
        x=x,y=subset['5% bound (s)'],mode='lines',fill='tonexty',
        fillcolor='rgba(100,149,237,0.2)',line=dict(width=0),
        name='Middle 90% of fitted Normal',hovertemplate='Lower 5% bound: %{y:.2f} s<extra></extra>'))
    fig.add_trace(go.Scatter(
        x=x,y=subset['Fitted μ (s)'],mode='lines+markers',
        name='Fitted μ (five preceding observations)'))
    fig.add_trace(go.Scatter(
        x=x,y=subset['Observed (s)'],mode='markers',
        marker=dict(size=11),name='Held-out observed duration'))
    fig.update_layout(
        title=f'{product} · {metric} · rolling five-observation model check',
        xaxis_title='Held-out historical observation number',
        yaxis_title='Processing duration (s)',
        xaxis=dict(dtick=1),height=390,margin=dict(l=30,r=20,t=45,b=30))
    st.plotly_chart(fig,use_container_width=True)
    st.caption('For each target, fit a positive-truncated Normal using ONLY the preceding five '
               'observations of this product. Then check where the held-out measured duration '
               'falls relative to the fitted distribution. This assesses a processing-time model; '
               'it is not a future-demand or completion-time forecast. '
               'The 90% interval is model-based, not a guarantee of 90% empirical coverage. '
               'With only four initial held-out observations per product/station, coverage is noisy.')
    with st.expander('Rolling calibration windows and held-out checks',expanded=True):
        st.dataframe(subset.round(3),hide_index=True,use_container_width=True)
    st.info('Scope: these checks validate observed workstation-duration distributions. '
            'They do NOT independently validate MC6–MC9 allocation shares, Weibull failure '
            'behavior, factory throughput, or makespan without corresponding observed ground truth.')
    with st.expander('Latest fetched raw observations'):
        st.dataframe(snap['df'].drop(columns=['row_sequence']),hide_index=True,use_container_width=True)


with tab_opt:
    O=snap['optimization']
    st.subheader('Simulation-based WIP and operational recommendations')
    st.caption('Finite requested order; time-based improvement only. No financial optimization or preventative maintenance schedules.')
    a,b,c=st.columns(3)
    a.metric('Recommended operating WIP',O['recommended_wip'])
    b.metric('Theoretical critical WIP (idealized)',f"{O['theoretical_wip']:.1f}")
    c.metric('Theoretical capacity-limiting stage',O['theoretical_bottleneck'])
    st.caption('Recommended WIP: smallest refined candidate reaching at least 95% of the best estimated finite-order throughput. '
               'Theoretical WIP ignores variability and assumes an idealized workstation model.')
    if O.get('wip_refinement_runs_used',30)<20:
        st.info('WIP estimates use a limited replication budget. Close alternatives may be indistinguishable.')
    if not O['saturation_supported']:
        st.warning('A convincing throughput plateau was not established within the tested WIP ceiling. Treat the recommendation as provisional.')
    wdata=O['wip_table'].sort_values('WIP')
    c1,c2=st.columns(2)
    with c1:
        f1=px.line(wdata,x='WIP',y='throughput_h',markers=True,title='Throughput versus global WIP',
                   labels={'throughput_h':'Mean throughput (units/hour)'})
        f1.add_vline(x=O['recommended_wip'],line_dash='dash')
        f1.update_layout(height=370)
        st.plotly_chart(f1,use_container_width=True)
    with c2:
        f2=px.line(wdata,x='WIP',y='cycle_min',markers=True,title='Cycle time versus global WIP',
                   labels={'cycle_min':'Mean product cycle time (minutes)'})
        f2.add_vline(x=O['recommended_wip'],line_dash='dash')
        f2.update_layout(height=370)
        st.plotly_chart(f2,use_container_width=True)
    st.subheader('Actual WIP versus configured WIP limit')
    fw=px.line(wdata,x='WIP',y='mean_wip',markers=True,
               labels={'mean_wip':'Mean actual WIP (products)','WIP':'Configured WIP limit'},
               title='Actual mean WIP: admission limit versus achieved work in process')
    fw.add_scatter(x=wdata['WIP'],y=wdata['max_actual_wip'],mode='lines+markers',
                   name='Mean of maximum actual WIP per run')
    fw.update_layout(height=310)
    st.plotly_chart(fw,use_container_width=True)
    st.caption('Increasing the global limit does not necessarily increase actual WIP: '
               'WS1 admission and downstream capacities may be binding. '
               'Peak WIP shown is the mean of each replication’s peak, not a single global maximum.')
    st.caption('Curve points use differing screening/refinement replication counts, shown in the table. '
               'The curve is for finite batches, not a continuous steady-state factory.')
    with st.expander('WIP curve data and Monte Carlo precision'):
        st.dataframe(wdata.round(4),hide_index=True,use_container_width=True)
    if O.get('estimated_total_optimizer_runs',0)*sum(snap['quantities'].values())>450000:
        st.warning('The online optimizer exceeded its heuristic product-run budget. '
                   'Runtime/precision are not guaranteed for this workload; use offline Python '
                   'for high-confidence annual-scale analysis.')
    st.subheader('Ranked single-intervention recommendations')
    st.caption('Standardized candidate actions: MTTF +25%/+50%; repair mean -25%/-50%; buffers +1/+2. '
               'Only the strongest screening candidates are rerun at higher precision. '
               'All comparisons use the recommended operating WIP, unchanged demand and other parameters.')
    recs=O['recommendations']
    meta=O.get('search_metadata',{})
    st.caption(f"Screened {meta.get('evaluated_candidates',0)} of {meta.get('total_candidates',54)} interventions; "
               f"{meta.get('screening_runs',0)} screening runs and "
               f"{meta.get('refinement_runs',0)} refinement runs per shortlisted option. "
               f"Approximately {O.get('estimated_total_optimizer_runs',0)} optimizer replications executed.")
    if meta.get('search_limited'):
        st.warning('Limited computational search: not all 54 intervention scenarios were tested. '
                   'Rankings cover ONLY the candidates actually screened; do not interpret this as a global optimum.')
    st.dataframe(recs.round(3),hide_index=True,use_container_width=True)
    if not recs.empty and (recs['Evidence']=='Uncertain').any():
        st.info('Uncertain means the paired 95% Monte Carlo interval includes zero improvement. '
                'A small numerical advantage is not proof of a real operational benefit.')
    if not recs.empty:
        st.caption('Mean time saved is a simulated paired point estimate. Its 95% CI quantifies Monte Carlo '
                   'uncertainty only, not uncertainty in measured processing or reliability inputs. '
                   'Top-ranked does not mean proven superior to all other candidates.')
        st.download_button('Download recommendations (CSV)',recs.to_csv(index=False).encode(),
                           file_name='factory_recommendations.csv',mime='text/csv')
    st.subheader('Bottleneck diagnostics — baseline at selected WIP')
    state_summary=S['state_table']
    d=state_summary[state_summary['State'].isin(['B3','B5','B6'])].pivot(
        index='Resource',columns='State',values='Mean share (%)').fillna(0)
    d=d.rename(columns={'B3':'Processing (%)','B5':'Maintenance (%)','B6':'Blocked (%)'})
    d['Mean queue (products)']=[float(np.mean([r['queue_mean'].get(m,0) for r in R])) for m in d.index]
    st.dataframe(d.round(2),use_container_width=True)
    st.caption('High utilization, blocking and downtime are diagnostics, not proof of the bottleneck. '
               'Compare intervention sensitivity before assigning a causal bottleneck label.')
    st.subheader('Little’s Law — internal consistency check')
    mean_lead=np.mean([sum(x['lead_sum'].values())/max(1,sum(x['lead_count'].values())) for x in R])/3600
    lhs=S['mean_wip']
    rhs=np.mean([x['throughput_h']*(sum(x['lead_sum'].values())/max(1,sum(x['lead_count'].values()))/3600) for x in R])
    st.write(f'Mean WIP = **{lhs:.3f}** products; mean of throughput × flow time per run = **{rhs:.3f}** products.')
    st.caption('Finite-batch conservation identity when all admitted products complete; this checks accounting, '
               'not empirical accuracy against a physical factory.')
