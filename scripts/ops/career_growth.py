import sqlite3, numpy as np, glob, re
snaps = sorted(glob.glob('data/career_snap/pieces_*.sqlite3'))
def load(p):
    c=sqlite3.connect(p); return {r[0]:(r[1],np.frombuffer(r[2],dtype=np.float32),*r[3:]) for r in c.execute('select piece_id,species,theta,games,survivals,promotions,mvp_count,contribution from pieces')}
D={int(re.search(r'(\d+)\.sqlite3',p).group(1)):load(p) for p in snaps}
ns=sorted(D); print('snap:', ns)
norms=np.array([np.mean([np.linalg.norm(v[1]) for v in D[n].values()]) for n in ns])
print('n     ', ' '.join(f'{n:5d}' for n in ns)); print('norm  ', ' '.join(f'{x:5.3f}' for x in norms))
# log-log 傾き: ランダムウォーク 0.5, 直進 1.0
x=np.log(ns[1:]); y=np.log(norms[1:]); s=np.polyfit(x,y,1)[0]; print(f'log-log 傾き (200局以降) {s:.2f}  [√n=0.50, 直進=1.00]')
x=np.log(ns[-8:]); y=np.log(norms[-8:]); print(f'log-log 傾き (直近8点) {np.polyfit(x,y,1)[0]:.2f}')
a,b=D[ns[0]],D[ns[-1]]
cos=[float(a[k][1]@b[k][1]/(np.linalg.norm(a[k][1])*np.linalg.norm(b[k][1])+1e-9)) for k in a]
print(f'cos(θ@{ns[0]}, θ@{ns[-1]}) 平均 {np.mean(cos):.2f}  >0.5: {np.mean(np.array(cos)>0.5):.2f}')
# 種別内コサイン + ブートストラップ CI
rng=np.random.default_rng(0); U={k:v[1]/(np.linalg.norm(v[1])+1e-9) for k,v in b.items()}
sp={}
for k,v in b.items(): sp.setdefault(v[0],[]).append(U[k])
print('種別  n  within-cos (最終)')
for s_,vs in sp.items():
    if len(vs)>1:
        X=np.stack(vs); m=X@X.T; off=m[~np.eye(len(vs),dtype=bool)]
        print(f'{s_:>4} {len(vs):>2}  {off.mean():+.2f}')
allv=np.stack(list(U.values())); M=allv@allv.T; print(f' 全ペア {M[~np.eye(len(M),dtype=bool)].mean():+.2f}')
n_=np.array([np.linalg.norm(v[1]) for v in b.values()]); pr=np.array([v[4] for v in b.values()]); ct=np.array([v[6] for v in b.values()]); sv=np.array([v[3]/max(v[2],1) for v in b.values()])
print(f'corr(norm, promotions) {np.corrcoef(n_,pr)[0,1]:+.2f}  corr(norm, contribution) {np.corrcoef(n_,ct)[0,1]:+.2f}  corr(norm, survival) {np.corrcoef(n_,sv)[0,1]:+.2f}')
