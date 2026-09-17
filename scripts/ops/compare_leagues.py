import json, numpy as np, sys
runs = sys.argv[1:] or ['league_run2_6x80','league_E1_control']
def stats(run, G=50):
    h=json.load(open(f'checkpoints/{run}/league_history.json'))[:G]; G=len(h); names=sorted(h[0]['win_rates'])
    W=np.array([[g['win_rates'][c] for c in names] for g in h]); d=np.array([g['phi_distance']['mean'] for g in h])
    sp=W.max(1)-W.min(1)
    sp_ex=[]; trap=0
    for t in range(1,G):
        prev=h[t-1]['renewed']          # drift (変異のみ) の run では空になる
        nb=prev[0] if prev else None
        wr={c:v for c,v in h[t]['win_rates'].items() if c!=nb}; sp_ex.append(max(wr.values())-min(wr.values()))
        trap+= nb is not None and nb in h[t]['renewed']
    rhos=[]
    for t in range(G-1):
        idx=[i for i,c in enumerate(names) if c not in h[t]['renewed']]
        rhos.append(np.corrcoef(W[t,idx].argsort().argsort(), W[t+1,idx].argsort().argsort())[0,1])
    draws=[g.get('draw_rate') for g in h if g.get('draw_rate') is not None]
    return dict(G=G, draw=(np.mean(draws) if draws else float('nan')), dist10=d[9], dist20=d[19], dist30=d[29], dist_end=d[G-1], dist_last10=d[-10:].mean(), dist_max=d.max(),
                spread=sp.mean(), spread_ex=np.mean(sp_ex), rho=np.nanmean(rhos), trap=f'{trap}/{G-1}',
                per_culture=' '.join(f'{W[:,i].mean():.2f}' for i in range(len(names))))
rows={r:stats(r) for r in runs}
keys=['G','draw','dist10','dist20','dist30','dist_end','dist_last10','dist_max','spread','spread_ex','rho','trap','per_culture']
print(f'{"":14}'+''.join(f'{r[7:]:>30}' for r in runs))
for k in keys:
    print(f'{k:14}'+''.join(f'{(f"{v[k]:.3f}" if isinstance(v[k],float) else str(v[k])):>30}' for v in rows.values()))
