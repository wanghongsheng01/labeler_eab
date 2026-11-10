"""Paired channel check (2026-11-10): the same GPT-6 requests (reasoning effort none) through the first gateway
(gpt-6-sol-nr, Block B on five datasets, 2026-11-09) and the second gateway (gpt-6-sol-nr2, 2026-11-10). Agreement of
the valid labels per tuple between the channels, against the agreement between the canonical calls and their rerun
within the second channel. Output: results/channel_check.csv"""
import glob, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{HERE}/results/main"
rows = []
for t in ["imdb", "yelp", "agnews", "emotion", "hate"]:
    a = pd.read_csv(f"{M}/B10__gpt-6-sol-nr__{t}.csv"); b = pd.read_csv(f"{M}/B10__gpt-6-sol-nr2__{t}.csv")
    r = pd.read_csv(f"{M}/R10__gpt-6-sol-nr2__{t}.csv")
    a = a[a.rep == 0].set_index(["order", "tid"]).pred; b = b[b.rep == 0].set_index(["order", "tid"]).pred
    j = pd.concat([a.rename("x"), b.rename("y")], axis=1).dropna(); j = j[(j.x >= 0) & (j.y >= 0)]
    c = b.xs("shuffled"); rr = r.set_index("tid").pred.reindex(c.index)
    ok = (c >= 0) & (rr >= 0)
    rows.append(dict(task=t, pairs=len(j), cross_channel_agree=float((j.x == j.y).mean()),
                     same_channel_rerun_agree=float((c[ok] == rr[ok]).mean())))
D = pd.DataFrame(rows); D.to_csv(f"{HERE}/results/channel_check.csv", index=False)
print(D.round(4).to_string())
print("pooled cross", round((D.cross_channel_agree * D.pairs).sum() / D.pairs.sum(), 4), "rerun mean", round(D.same_channel_rerun_agree.mean(), 4))

# Second pair (2026-11-10): GLM-5.3 through the gateway (glm-5.3-gw, thinking disabled but still reasoning) and through
# Zhipu's own endpoint (glm-5.3-zp, thinking disabled, no reasoning tokens): identical Block L requests (rep 0 of every
# order), against the rerun agreement within each channel. Output: results/channel_check_glm.csv
rows = []
for t in ["agnews", "emotion", "yelp", "hate"]:
    a, b = (pd.read_csv(f"{M}/L__{m}__{t}.csv") for m in ("glm-5.3-gw", "glm-5.3-zp"))
    x = a[a.rep == 0].set_index(["order", "tid"]).pred; y = b[b.rep == 0].set_index(["order", "tid"]).pred
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna(); j = j[(j.x >= 0) & (j.y >= 0)]
    r = dict(task=t, pairs=len(j), cross_channel_agree=float((j.x == j.y).mean()))
    for m, d in (("gw", a), ("api", b)):
        c = d[(d.order == "shuffled") & (d.rep == 0)].set_index("tid").pred
        rr = d[(d.order == "shuffled") & (d.rep == 1)].set_index("tid").pred.reindex(c.index)
        ok = (c >= 0) & (rr >= 0)
        r[f"rerun_agree_{m}"] = float((c[ok] == rr[ok]).mean())
        r[f"acc_{m}"] = float((d[(d.order == "shuffled") & (d.rep == 0)].pred == d[(d.order == "shuffled") & (d.rep == 0)].gold).mean())
    rows.append(r)
G = pd.DataFrame(rows); G.to_csv(f"{HERE}/results/channel_check_glm.csv", index=False)
print(G.round(4).to_string())
print("GLM pooled cross", round((G.cross_channel_agree * G.pairs).sum() / G.pairs.sum(), 4),
      "rerun gw", round(G.rerun_agree_gw.mean(), 4), "rerun api", round(G.rerun_agree_api.mean(), 4))
