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


def querylevel():
    q = pd.read_csv(f"{R}/main_querylevel.csv").query("B == 10")
    lines = [r"\begin{tabular}{@{}l" + "r" * 8 + "@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{Tuples, $\geq$2 labels (\%)} & \multicolumn{4}{c}{Worst Jaccard (group/rerun)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
             "Dataset & " + " & ".join(MN[m] for m in MODELS) + " & " + " & ".join(MN[m] for m in MODELS) + r" \\",
             r"\midrule"]
    for t in TASKS:
        x = q[q.task == t].set_index("model").reindex(MODELS)
        lines.append(TN[t] + " & " + " & ".join(f"{100 * v:.1f}" for v in x.unstable_any) + " & " +
                     " & ".join(f"{v:.2f}/{r:.2f}".replace("0.", ".").replace("1.00", "1") for v, r in zip(x.jac_min, x.jac_rerun_min)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
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
    for t in TASKS:
        cells = []
        for m in ms:
            r = d[(d.model == m) & (d.task == t)].iloc[0]
            cells += [star(r.format_pp, r.format_p, r.format_p_holm), star(r.context50_pp, r.context50_p, r.context50_p_holm)]
        lines.append(TN[t] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{T}/decomposition.tex", "w").write("\n".join(lines) + "\n")


def xvendor():
    """Table of the labelers added for vendor coverage (analyze_xvendor.py)."""
    f = f"{R}/xv_summary.csv"
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    XN = {"gpt-6-sol-nr": "GPT-6", "gpt-6-sol": "GPT-6 (default reasoning)", "claude-opus-5.5": "Claude", "qwen2.5-3b": "Qwen-3B", "qwen2.5-7b": "Qwen-7B"}
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


if __name__ == "__main__":
    xvendor()
    querylevel(); decomposition(); triggers()
    print("ok")
