# B31 pre-registration diagnostic, run on the local CPU 2026-11-18 ~05:10 UTC (arena A train + dev only). Output: a_exact.out
# Diagnostic (train + dev of arena A only): exact-symbol HRR chain cosines, comm vs perm; published learned readout bound vs closed-form threshold.
import sys, re, random, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/path/to/vsassm"); sys.path.insert(0, "/path/to/vsassm/b27_arena")
import b27_common as C
class Cfg: num_roles=64; vocab_size=256; num_pairs=16; max_length=512; p_positive=0.5
MK = re.compile(r"\[([RXY])([0-9A-F]{2})\]"); FL = re.compile(r"\{(\d+)\}")
def parse(text):
    a, rest = text[2:].split(" B ", 1); b, rest = rest.split(" C ", 1); c, q = rest.split(" Q ", 1)
    A = [(int(x,16), int(y,16)) for (k1,x),(k2,y) in zip(MK.findall(a)[0::2], MK.findall(a)[1::2])]
    B = [(int(x,16), int(y,16)) for (k1,x),(k2,y) in zip(MK.findall(b)[0::2], MK.findall(b)[1::2])]
    C_ = list(zip([int(y,16) for k,y in MK.findall(c)], [int(f) for f in FL.findall(c)]))
    qr = int(MK.findall(q)[0][1],16); cand = int(FL.findall(q)[0])
    return A, B, C_, qr, cand
def unit(n,d,s):
    g=torch.Generator(); g.manual_seed(s); v=torch.randn(n,d,generator=g); return v/(v.norm(dim=-1,keepdim=True)+1e-8)
def bind(a,b): return torch.fft.ifft(torch.fft.fft(a.float())*torch.fft.fft(b.float())).real
def unbind(ab,a): return torch.fft.ifft(torch.fft.fft(ab.float())*torch.conj(torch.fft.fft(a.float()))).real
def l2(x): return x/(x.norm(dim=-1,keepdim=True)+1e-8)
def bundle(v): s=v.sum(0); return s/(s.norm(dim=-1,keepdim=True)+1e-8)
for seed in (42,123,456):
    torch.manual_seed(seed); R = torch.randn(64,2048); R = R/R.norm(dim=-1,keepdim=True)
    X, Y, Fv = unit(256,2048,seed+11), unit(256,2048,seed+23), unit(256,2048,seed+37)
    P = torch.randperm(2048, generator=torch.Generator().manual_seed(seed+7001))
    for split, n, off in (("train",2000,42),("dev",500,777)):
        for enc in ("comm","perm"):
            V = (lambda M: M[:,P]) if enc=="perm" else (lambda M: M)
            Xv, Yv, Fvv = V(X), V(Y), V(Fv)
            sims, ys, ok = [], [], []
            for i in range(n):
                text, y = C.gen_chain(i, off, Cfg)
                A,B,C_,qr,cand = parse(text)
                memA = bundle(torch.stack([bind(Xv[x], R[r]) for r,x in A]))
                xh = l2(unbind(memA, R[qr])); xs=[x for _,x in A]
                xw = xs[int(F.cosine_similarity(Xv[xs], xh[None]).argmax())]
                memB = bundle(torch.stack([bind(Yv[yy], X[x]) for x,yy in B]))
                yh = l2(unbind(memB, X[xw])); yss=[yy for _,yy in B]
                yw = yss[int(F.cosine_similarity(Yv[yss], yh[None]).argmax())]
                memC = bundle(torch.stack([bind(Fvv[f], Y[yy]) for yy,f in C_]))
                fh = l2(unbind(memC, Y[yw]))
                s = float(F.cosine_similarity(fh[None], Fvv[cand][None]))
                d = {r:x for r,x in A}; d2={x:yy for x,yy in B}; d3={yy:f for yy,f in C_}
                ok.append(int(xw==d[qr] and yw==d2[d[qr]])); sims.append(s); ys.append(y)
                assert int(d3[d2[d[qr]]]==cand)==y
            s=np.array(sims); ys=np.array(ys)
            acc20 = ((s>0.2)==(ys==1)).mean()
            c=np.unique(s); accs=[((s>=t)==(ys==1)).mean() for t in c]; best=max(accs)
            print(seed, split, enc, "retrieval ok %.4f"%np.mean(ok), "pos sim mean %.3f min %.3f p05 %.3f | neg max %.3f"%(s[ys==1].mean(), s[ys==1].min(), np.percentile(s[ys==1],5), s[ys==0].max()),
                  "acc@0.20 %.4f  best-threshold acc %.4f"%(acc20, best), flush=True)
