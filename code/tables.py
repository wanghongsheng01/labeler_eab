"""LaTeX tables of the paper, generated from results/*.csv so that no number is transcribed by hand.
Writes paper/tables/*.tex."""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, T = f"{HERE}/results", f"{HERE}/paper/tables"
os.makedirs(T, exist_ok=True)
TASKS = ["imdb", "agnews", "offensive", "yelp", "amazon", "emotion", "hate"]
TN = {"imdb": "IMDB", "agnews": "AG News", "offensive": "Offensive", "yelp": "Yelp", "amazon": "Amazon",
      "emotion": "Emotion", "hate": "Hate"}
MODELS = ["ds-flash", "ds-pro", "glm-5.3-flash", "kimi-k2.6"]
MN = {"ds-flash": "DS-Flash", "ds-pro": "DS-Pro", "glm-5.3-flash": "GLM", "kimi-k2.6": "Kimi"}


def rank_marks(vals, high):
    """Bold the best printed value, underline the second best (ties share the mark)."""
    u = sorted(set(vals), reverse=high)
    return ["b" if v == u[0] else "u" if len(u) > 1 and v == u[1] else "" for v in vals]


def wrap(txt, k):
    return r"\textbf{" + txt + "}" if k == "b" else r"\underline{" + txt + "}" if k == "u" else txt


def querylevel():
    # per dataset: fewest tuples with two labels, and the highest worst-class Jaccard across groupings, are best
    q = pd.read_csv(f"{R}/main_querylevel.csv").query("B == 10")
    lines = [r"\begin{tabular}{@{}l" + "r" * 8 + "@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{Tuples, $\geq$2 labels (\%)} & \multicolumn{4}{c}{Worst Jaccard (group/rerun)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
             "Dataset & " + " & ".join(MN[m] for m in MODELS) + " & " + " & ".join(MN[m] for m in MODELS) + r" \\",
             r"\midrule"]
    cnt = {m: [0, 0, 0, 0] for m in MODELS}
    for t in TASKS:
        x = q[q.task == t].set_index("model").reindex(MODELS)
        u = [round(100 * v, 1) for v in x.unstable_any]
        jg = [round(v, 2) for v in x.jac_min]
        mu, mj = rank_marks(u, False), rank_marks(jg, True)
        for m, a, b in zip(MODELS, mu, mj):
            cnt[m][0] += a == "b"; cnt[m][1] += a == "u"; cnt[m][2] += b == "b"; cnt[m][3] += b == "u"
        lines.append(TN[t] + " & " + " & ".join(wrap(f"{v:.1f}", k) for v, k in zip(u, mu)) + " & " +
                     " & ".join(wrap(f"{v:.2f}".replace("0.", ".").replace("1.00", "1"), k) + f"/{r:.2f}".replace("0.", ".").replace("1.00", "1")
                                for v, k, r in zip(jg, mj, x.jac_rerun_min)) + r" \\")
    lines += [r"\midrule", "Best/2nd & " + " & ".join(f"{cnt[m][0]}/{cnt[m][1]}" for m in MODELS) + " & " +
              " & ".join(f"{cnt[m][2]}/{cnt[m][3]}" for m in MODELS) + r" \\", r"\bottomrule", r"\end{tabular}"]
    open(f"{T}/querylevel.tex", "w").write("\n".join(lines) + "\n")


def decomposition():
    # effects and tests of analyze_revision.py: format = exact McNemar (one text per call), context = z-test with
    # standard errors clustered by call; Holm within each family of 28 contrasts. All four planned labelers.
    d = pd.read_csv(f"{R}/rev_decomposition.csv")
    sgn = lambda v: f"{v:+.1f}".replace("-", "$-$")
    star = lambda v, p, ph: sgn(v) + (r"$^{**}$" if ph < 0.05 else r"$^{*}$" if p < 0.05 else "")
    ms = ["ds-flash", "ds-pro", "glm-5.3-flash", "kimi-k2.6"]
    lines = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{2}{c}{" + MN[m] + "}" for m in ms) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(4)),
             "Dataset" + " & Fmt. & Ctx." * 4 + r" \\", r"\midrule"]
    cnt = {m: [0, 0, 0, 0] for m in ms}
    for t in TASKS:
        rr = [d[(d.model == m) & (d.task == t)].iloc[0] for m in ms]
        mf = rank_marks([round(r.format_pp, 1) for r in rr], True)
        mc = rank_marks([round(r.context50_pp, 1) for r in rr], True)
        cells = []
        for m, r, a, b in zip(ms, rr, mf, mc):
            cnt[m][0] += a == "b"; cnt[m][1] += a == "u"; cnt[m][2] += b == "b"; cnt[m][3] += b == "u"
            cells += [wrap(star(r.format_pp, r.format_p, r.format_p_holm), a), wrap(star(r.context50_pp, r.context50_p, r.context50_p_holm), b)]
        lines.append(TN[t] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", "Best/2nd & " + " & ".join(f"{cnt[m][0]}/{cnt[m][1]} & {cnt[m][2]}/{cnt[m][3]}" for m in ms) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    open(f"{T}/decomposition.tex", "w").write("\n".join(lines) + "\n")


def xvendor():
    """Table of the labelers added for vendor coverage (analyze_xvendor.py)."""
    f = f"{R}/xv_summary.csv"
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    XN = {"gpt-6-sol-nr2": "GPT-6", "gpt-6-sol-nr": "GPT-6 (gw-1)", "gpt-6-sol": "GPT-6 (default reasoning)", "claude-opus-5.5": "Claude", "qwen2.5-3b": "Qwen-3B", "qwen2.5-7b": "Qwen-7B"}
    pct = lambda v: "--" if pd.isna(v) else f"{100 * v:.1f}"
    sgn = lambda v: "--" if pd.isna(v) else f"{v:+.1f}".replace("-", "$-$")
    star = lambda v, p: sgn(v) + (r"$^{*}$" if (not pd.isna(p)) and p < 0.05 else "")
    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"Labeler & Dataset & Acc. & CS & RR & Perm. & Sorted & Keyed \\", r"\midrule"]
    first = True
    for m in XN:
        x = d[d.model == m].set_index("task").reindex([t for t in TASKS if t in set(d[d.model == m].task)])
        if not len(x): continue
        if not first: lines.append(r"\midrule")
        first = False
        for i, (t, r) in enumerate(x.iterrows()):
            lines.append((XN[m] if i == 0 else "") + f" & {TN[t]} & {pct(r.acc)} & {pct(r.comp)} & {pct(r.rerun)} & "
                         f"{pct(r.get('pos'))} & {star(r.sorted_pp, r.sorted_p_holm)} & {star(r.keyed_pp, r.keyed_p_holm)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{T}/xvendor.tex", "w").write("\n".join(lines) + "\n")


def triggers():
    d = pd.read_csv(f"{R}/main/D__decode.csv")
    lines = [r"\begin{tabular}{@{}lrlrl@{}}", r"\toprule",
             r"Dataset & Tuple & Gold & Refused & Topic (paraphrased) \\", r"\midrule"]
    topic = {1525: "China's terms for Taiwan talks", 1895: "Taiwan's president vs.\\ opposition",
             165: "China warns Singapore on Taiwan", 957: "Taiwan urges talks with China",
             1680: "sexually abusive insult", 1424: "sexually violent threat"}
    for r in d.itertuples():
        lab = {"agnews": ["World", "Sports", "Business", "Sci/Tech"], "hate": ["not hateful", "hateful"]}[r.task][r.gold]
        lines.append(f"{TN[r.task]} & {r.tid} & {lab} & {r.n_refused}/{r.n_batches} & {topic[r.tid]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{T}/triggers.tex", "w").write("\n".join(lines) + "\n")



def alltable():
    """Table VI: every labeler on the four tasks that all of them ran (analyze_xvendor.py all). Lower is better in every
    column; per column the lowest value is bold and the second lowest underlined (ties share the mark), and the last
    column counts each labeler's marks."""
    f = f"{R}/all_summary.csv"
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    NM = {"ds-flash": "DS-Flash", "ds-pro": "DS-Pro", "glm-5.3-flash": "GLM-5.3-Flash", "kimi-k2.6": "Kimi-K2.6",
          "gpt-6-sol-nr2": "GPT-6", "gpt-6.1-sol": "GPT-6.1", "gpt-6-astra": "GPT-6-Astra", "claude-opus-5.5": "Claude",
          "kimi-k3-gw": "Kimi-K3", "glm-5.3-gw": "GLM-5.3 (gw)", "kimi-k3.1-ark": "Kimi-K3.1 (Ark)",
          "glm-5.3-zp": "GLM-5.3 (API)", "doubao-seed-2.1": "Doubao-Seed-2.1", "qwen3.8-27b": "Qwen3.8-27B", "qwen2.5-3b": "Qwen-3B",
          "qwen2.5-7b": "Qwen-7B"}
    T4 = ["agnews", "emotion", "yelp", "hate"]
    rows = []
    for m in NM:
        x = d[(d.model == m) & d.task.isin(T4)].set_index("task")
        if len(x) < 4:
            continue
        r = {"m": m}
        for t in T4:
            r[t] = round(100 * x.loc[t, "comp"], 1)
        r["rr"] = round(100 * x.rerun.mean(), 1)
        r["sorted"] = round(x.sorted_pp.abs().mean(), 1)
        rows.append(r)
    D = pd.DataFrame(rows)
    cols = T4 + ["rr", "sorted"]
    mark = {c: {} for c in cols}
    for c in cols:
        vals = sorted(set(D[c]))
        for i, row in D.iterrows():
            mark[c][i] = "b" if row[c] == vals[0] else "u" if len(vals) > 1 and row[c] == vals[1] else ""
    fmt = lambda v, k: (r"\textbf{%.1f}" % v) if k == "b" else (r"\underline{%.1f}" % v) if k == "u" else "%.1f" % v
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{CS (\%)} & & & \\", r"\cmidrule(lr){2-5}",
             r"Labeler & AG & Emo. & Yelp & Hate & RR & $|$Sort.$|$ & B/2 \\", r"\midrule"]
    groups = [("ds-flash", "ds-pro", "glm-5.3-flash", "kimi-k2.6"),
              ("gpt-6-sol-nr2", "gpt-6.1-sol", "gpt-6-astra", "claude-opus-5.5", "kimi-k3-gw", "glm-5.3-gw", "kimi-k3.1-ark", "glm-5.3-zp", "doubao-seed-2.1"),
              ("qwen3.8-27b", "qwen2.5-3b", "qwen2.5-7b")]
    for gi, g in enumerate(groups):
        for i, row in D.iterrows():
            if row.m not in g:
                continue
            nb = sum(mark[c][i] == "b" for c in cols); nu = sum(mark[c][i] == "u" for c in cols)
            lines.append(NM[row.m] + " & " + " & ".join(fmt(row[c], mark[c][i]) for c in cols) + f" & {nb}/{nu} \\\\")
        if gi < len(groups) - 1 and any(m in set(D.m) for m in groups[gi + 1]):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{T}/alltable.tex", "w").write("\n".join(lines) + "\n")
    D.to_csv(f"{R}/all_table.csv", index=False)

if __name__ == "__main__":
    xvendor(); alltable()
    querylevel(); decomposition(); triggers()
    print("ok")
