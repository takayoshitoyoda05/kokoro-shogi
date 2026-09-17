"""最終文化の距離行列 + 系統(2クラスタ)比 + レーダー(欲求6軸, base比%)."""
import sys, json, itertools
import numpy as np, torch
AX = ("survive", "attack", "promote", "defend", "advance", "redeploy")
base = torch.load("checkpoints/ppo2.pt", map_location="cpu", weights_only=False)
bs = base["model"]
W = bs["personality.project.weight"][:, :16].numpy(); b = bs["personality.project.bias"].numpy()
def head(th):
    z = th.reshape(-1, 16) @ W.T + b
    return np.log1p(np.exp(z)) + 1e-4
def theta_of(st):
    for k, v in st.items():
        if "theta_sp" in k or k.endswith("theta_species"):
            return v.numpy().reshape(-1)
    raise KeyError([k for k in st if "theta" in k])
base_th = theta_of(bs)
for name in sys.argv[1:]:
    L = torch.load(f"checkpoints/{name}/league.pt", map_location="cpu", weights_only=False)
    ths = {n: c["theta_sp"].numpy().reshape(-1) for n, c in L["cultures"].items()}
    names = list(ths)
    D = np.array([[np.linalg.norm(ths[a]-ths[b_]) for b_ in names] for a in names])
    # 2クラスタ: 最遠ペアを種にして近い方へ割当
    i, j = np.unravel_index(D.argmax(), D.shape)
    A = [n for k, n in enumerate(names) if D[k, i] <= D[k, j]]
    B = [n for n in names if n not in A]
    within = [D[names.index(x), names.index(y)] for g in (A, B) for x, y in itertools.combinations(g, 2)]
    between = [D[names.index(x), names.index(y)] for x in A for y in B]
    print(f"\n== {name} ==  A={A} B={B}")
    print(f"within {np.mean(within):.2f} (n={len(within)})  between {np.mean(between):.2f}  ratio {np.mean(between)/max(np.mean(within),1e-9):.2f}")
    print("dist to base:", {n: round(float(np.linalg.norm(t-base_th)), 2) for n, t in ths.items()})
    hb = head(base_th).reshape(-1)
    npc = hb.size // len(AX)
    hb = hb.reshape(npc, len(AX)).mean(0)
    print("axis%   " + " ".join(f"{a:>9}" for a in AX))
    for n in names:
        h = head(ths[n]).reshape(npc, len(AX)).mean(0)
        print(f"{n:8s}" + " ".join(f"{100*(h[k]/hb[k]-1):+8.1f}%" for k in range(len(AX))))
