"""Audited deposit NII/margin Shapley attribution, including CCAR cycle-over-cycle.

Install: pip install numpy pandas matplotlib

Pair:
 python attribution_v2.py --input sample_deposits.csv --mode pair --a Baseline --b Severe
Cycle:
 python attribution_v2.py --input sample_deposits_cycles.csv --mode cycle \
   --prior-cycle 2025 --current-cycle 2026 --base Baseline --stress Severe

Caveat: blended FTP yield is not an identified product-level FTP repricing effect.
"""
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLUMN_MAP = {
    'period': 'month', 'scenario': 'scenario', 'cycle': 'cycle',
    'ftp': 'ftp_dollars', 'expense': 'interest_expense_dollars',
    'total': 'total_avg_balance', 'dda': 'dda_avg_balance',
    'ib': 'ib_avg_balance', 'fed_funds': 'fed_funds_rate',
}
FACTORS = ('Balance', 'Mix', 'FTP yield', 'IB paid rate')
COLORS = ['#3274A1','#9064A1','#C48736','#32866C']


def load_data(path, day_count='fixed', mapping=COLUMN_MAP):
    raw = pd.read_csv(path)
    mandatory = ['period','scenario','ftp','expense','total','dda','ib']
    missing = [mapping[k] for k in mandatory if mapping[k] not in raw]
    if missing: raise ValueError(f'Missing required columns: {missing}')
    optional = [k for k in ('cycle','fed_funds') if mapping[k] in raw]
    x = raw[[mapping[k] for k in mandatory+optional]].rename(
        columns={mapping[k]:k for k in mandatory+optional})
    x['period'] = pd.to_datetime(x['period'], errors='raise')
    if x.period.isna().any(): raise ValueError('Missing periods')
    if not x.period.dt.is_month_start.all(): raise ValueError('Monthly dates must be first day of month')
    for k in ('ftp','expense','total','dda','ib'):
        x[k]=pd.to_numeric(x[k],errors='raise')
    if x[['ftp','expense','total','dda','ib']].isna().any().any():
        raise ValueError('Missing numeric values')
    if (x[['total','ib']]<=0).any().any() or (x.dda<0).any():
        raise ValueError('Total/IB must be positive; DDA nonnegative')
    if (x.expense<0).any(): raise ValueError('Interest expense must be a positive dollar cost')
    if ((x.total-x.dda-x.ib).abs()>np.maximum(1e-6, x.total.abs()*1e-6)).any():
        raise ValueError('Total average balance must equal DDA + IB')
    if 'cycle' not in x: x['cycle']='single'
    x['cycle']=x['cycle'].astype(str)
    key=['cycle','scenario','period']
    if x.duplicated(key).any(): raise ValueError('Duplicate cycle/scenario/month rows')
    x=x.sort_values(key).reset_index(drop=True)
    if day_count=='fixed':
        x['tau']=1/12  # 27 periods -> precisely 2.25 years, consistent with report shortcut
    elif day_count=='actual':
        x['tau']=x.period.dt.days_in_month/365
    else: raise ValueError('day_count must be fixed or actual')
    x['mix']=x.dda/x.total
    x['ftp_yield']=x.ftp/(x.total*x.tau)
    x['paid_yield']=x.expense/(x.ib*x.tau)
    x['nii']=x.ftp-x.expense
    return x


def factor_arrays(d):
    return {'Balance':d.total.to_numpy(float), 'Mix':d.mix.to_numpy(float),
            'FTP yield':d.ftp_yield.to_numpy(float),
            'IB paid rate':d.paid_yield.to_numpy(float)}


def evaluate(state,tau):
    b=state['Balance']; w=state['Mix']; f=state['FTP yield']; r=state['IB paid rate']
    exposure=b*tau
    monthly=exposure*(f-(1-w)*r)
    return monthly, float(monthly.sum()/exposure.sum())


def shapley(A,B,tau):
    n={k:np.zeros(len(tau)) for k in FACTORS}
    m={k:0.0 for k in FACTORS}
    for order in itertools.permutations(FACTORS):
        state=dict(A)
        prev_nii,prev_margin=evaluate(state,tau)
        for k in order:
            state[k]=B[k]
            next_nii,next_margin=evaluate(state,tau)
            n[k]+=(next_nii-prev_nii)/24
            m[k]+=(next_margin-prev_margin)*10000/24
            prev_nii,prev_margin=next_nii,next_margin
    return n,m


def selected(x,cycle,scenario):
    d=x[(x.cycle==str(cycle))&(x.scenario==scenario)].sort_values('period').reset_index(drop=True)
    if d.empty: raise ValueError(f'No data for cycle={cycle}, scenario={scenario}')
    return d


def compare(a,b,allow_calendar_shift=False):
    if len(a)!=len(b): raise ValueError('Both paths must have same number of monthly observations')
    if not allow_calendar_shift and not a.period.equals(b.period):
        raise ValueError('Comparison requires matching calendar months')
    if not np.allclose(a.tau.to_numpy(),b.tau.to_numpy()):
        raise ValueError('Accrual fractions differ; use fixed 1/12 for relative-horizon comparison')
    if len(a)!=27: print('NOTE: comparison includes',len(a),'months (not 27)')
    tau=a.tau.to_numpy()
    A=factor_arrays(a);B=factor_arrays(b)
    na,ma=evaluate(A,tau); nb,mb=evaluate(B,tau)
    contributions_n,contributions_m=shapley(A,B,tau)
    result={
        'nii_a':float(na.sum()),'nii_b':float(nb.sum()),
        'margin_a':ma*10000,'margin_b':mb*10000,
        'n':{k:float(contributions_n[k].sum()) for k in FACTORS},
        'm':contributions_m,
        'monthly':pd.DataFrame({'month_index':np.arange(1,len(a)+1),'nii_a':na,'nii_b':nb,
                                **{f'{k}_nii':contributions_n[k] for k in FACTORS}}),
        'exposure_a':float((a.total*a.tau).sum()),
        'exposure_b':float((b.total*b.tau).sum()),
    }
    assert np.isclose(sum(result['n'].values()),result['nii_b']-result['nii_a'],atol=1e-7,rtol=0)
    assert np.isclose(sum(result['m'].values()),result['margin_b']-result['margin_a'],atol=1e-8,rtol=0)
    assert np.allclose(result['monthly'][[f'{k}_nii' for k in FACTORS]].sum(axis=1),nb-na,atol=1e-7)
    return result


def plot_waterfall(path,title,start,contributions,unit):
    keys=list(contributions)
    values=np.array([contributions[k] for k in keys])
    fig,ax=plt.subplots(figsize=(10.5,4.7))
    ax.bar(0,start,color='#30485e',width=.68)
    current=start
    for i,(k,v) in enumerate(zip(keys,values),start=1):
        ax.bar(i,abs(v),bottom=min(current,current+v),color=COLORS[i-1],width=.68)
        ax.text(i,current+v/2,f'{v:+,.1f}',ha='center',va='center',fontsize=9,fontweight='bold')
        current+=v
    ax.bar(len(keys)+1,current,color='#30485e',width=.68)
    ax.set_xticks(range(len(keys)+2),['Reference',*keys,'Comparison'])
    ax.set_ylabel(unit)
    ax.set_title(title,loc='left',fontweight='bold',fontsize=14)
    ax.axhline(0,lw=.7,color='gray');ax.grid(axis='y',alpha=.12)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)


def save_result(out,prefix,result,description):
    tab=pd.DataFrame({'factor':FACTORS,'nii_dollars':[result['n'][k] for k in FACTORS],
                      'margin_bps':[result['m'][k] for k in FACTORS]})
    tab.to_csv(out/f'{prefix}_attribution.csv',index=False)
    result['monthly'].to_csv(out/f'{prefix}_monthly.csv',index=False)
    plot_waterfall(out/f'{prefix}_nii.png',description+' — NII',result['nii_a'],result['n'],'NII ($ input units)')
    plot_waterfall(out/f'{prefix}_margin.png',description+' — margin',result['margin_a'],result['m'],'Annualized bps')
    return tab


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='sample_deposits.csv')
    p.add_argument('--out',default='output_v2')
    p.add_argument('--mode',choices=['pair','cycle'],default='pair')
    p.add_argument('--day-count',choices=['fixed','actual'],default='fixed')
    p.add_argument('--a',default='Baseline');p.add_argument('--b',default='Severe')
    p.add_argument('--cycle',default='single')
    p.add_argument('--prior-cycle',default='2025');p.add_argument('--current-cycle',default='2026')
    p.add_argument('--base',default='Baseline');p.add_argument('--stress',default='Severe')
    args=p.parse_args()
    x=load_data(args.input,args.day_count)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.mode=='pair':
        a=selected(x,args.cycle,args.a);b=selected(x,args.cycle,args.b)
        r=compare(a,b)
        print(save_result(out,'pair',r,f'{args.b} − {args.a}'))
        print('NII diff:',r['nii_b']-r['nii_a'],'Margin diff bps:',r['margin_b']-r['margin_a'])
    else:
        prior=compare(selected(x,args.prior_cycle,args.base),selected(x,args.prior_cycle,args.stress))
        current=compare(selected(x,args.current_cycle,args.base),selected(x,args.current_cycle,args.stress))
        save_result(out,'prior_stress_minus_base',prior,'Prior cycle stress − base')
        save_result(out,'current_stress_minus_base',current,'Current cycle stress − base')
        difference={
            'nii_a':prior['nii_b']-prior['nii_a'],
            'nii_b':current['nii_b']-current['nii_a'],
            'margin_a':prior['margin_b']-prior['margin_a'],
            'margin_b':current['margin_b']-current['margin_a'],
            'n':{k:current['n'][k]-prior['n'][k] for k in FACTORS},
            'm':{k:current['m'][k]-prior['m'][k] for k in FACTORS},
            'monthly':current['monthly'].copy(),
        }
        difference['monthly']=pd.DataFrame({'month_index':prior['monthly'].month_index,
            **{f'{k}_nii':current['monthly'][f'{k}_nii']-prior['monthly'][f'{k}_nii'] for k in FACTORS}})
        assert np.isclose(sum(difference['n'].values()),difference['nii_b']-difference['nii_a'],atol=1e-7)
        assert np.isclose(sum(difference['m'].values()),difference['margin_b']-difference['margin_a'],atol=1e-8)
        tab=save_result(out,'cycle_change',difference,'Change in stress impact: current − prior')
        stress_prior=prior['nii_b']-prior['nii_a'];stress_current=current['nii_b']-current['nii_a']
        # Normalize with baseline deposit exposure from each cycle; bps per annum.
        normalized_prior=10000*stress_prior/prior['exposure_a']
        normalized_current=10000*stress_current/current['exposure_a']
        summary=pd.DataFrame([
            {'cycle':args.prior_cycle,'stress_nii_dollars':stress_prior,
             'stress_margin_bps':prior['margin_b']-prior['margin_a'],
             'baseline_exposure_dollar_years':prior['exposure_a'],
             'normalized_stress_nii_bps':normalized_prior},
            {'cycle':args.current_cycle,'stress_nii_dollars':stress_current,
             'stress_margin_bps':current['margin_b']-current['margin_a'],
             'baseline_exposure_dollar_years':current['exposure_a'],
             'normalized_stress_nii_bps':normalized_current},
        ])
        summary.to_csv(out/'cycle_summary.csv',index=False)
        print(tab.to_string(index=False));print(summary.to_string(index=False))
        print('NOTE: cycle factor differences compare two independently reconciled Shapley bridges.')

if __name__=='__main__': main()
