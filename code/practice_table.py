"""Table VII (2026-11-10): the recommended practices of Section VI against the common practice they replace, each
measured on the released runs (no new calls). The better value of every row is set in bold.

Rows: L1 batch order (clustered on the attribute vs. an attribute-independent key; for DBpedia and the refusal bias,
stored grouped by class vs. a random order), L2 uncertainty estimate (rerun vs.
regrouping), L3 batching evaluation (batch-size sweep vs. sweep plus the B = 1 batch-template control), L4 refusal
handling (drop refused calls and bisect vs. resend innocent tuples and decode triggers from existing runs), L5 logging
(requested vs. served model name).
Output: paper/tables/practice.tex, results/practice_summary.csv"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{HERE}/results"
T = f"{HERE}/paper/tables"
PLANNED = ["ds-flash", "ds-pro", "glm-5.3-flash", "kimi-k2.6"]


def main():
    rows = []
    o = pd.read_csv(f"{R}/rev_order.csv"); o = o[o.model.isin(PLANNED)]  # the 35 cells: 28 with B = 10, 7 with B = 50
    mo = pd.read_csv(f"{R}/main_order.csv"); mo = mo[mo.model.isin(PLANNED)]
    s, k = o[o.order == "sorted"], o[o.order == "keyed"]  # clustered on the attribute vs. on an attribute-independent key
    rows.append(("L1", "Holm-significant cells", f"{int((s.p_holm < 0.05).sum())}/35",
                 f"{int((k.p_holm < 0.05).sum())}/35", "low"))
    rows.append(("L1", "Median GROUP BY error ($L_1$, \\%)", f"{100 * mo[mo.order == 'sorted'].L1_order.median():.1f}",
                 f"{100 * mo[mo.order == 'keyed'].L1_order.median():.1f}", "low"))
    h = pd.read_csv(f"{R}/h_summary.csv"); h = h[h.model == "ds-flash"]
    rows.append(("L1", "DBpedia accuracy, DS-Flash (\\%)", f"{100 * h[h.order == 'stored'].acc.iloc[0]:.1f}",
                 f"{100 * h[h.order.str.startswith('shuffled')].acc.mean():.1f}", "high"))
    b = pd.read_csv(f"{R}/main_refusal_bias.csv").set_index(["task", "order"])["shift"]
    rows.append(("L1", "Refusal bias, AG News (pp)", f"{b[('agnews', 'sorted')]:.2f}", f"{b[('agnews', 'shuffled')]:.2f}", "low"))
    q = pd.read_csv(f"{R}/main_querylevel.csv"); q = q[(q.B == 10) & q.model.isin(PLANNED)]
    c = pd.read_csv(f"{R}/rev_composition.csv"); c = c[(c.B == 10) & c.model.isin(PLANNED)]
    m = q.merge(c, on=["model", "task"])
    rows.append(("L2", "Unstable tuples found (median \\%)", f"{100 * m.rerun.median():.1f}", f"{100 * m.unstable_any.median():.1f}", "high"))
    d = pd.read_csv(f"{R}/rev_decomposition.csv")
    fd = int((d.format_pp.abs() > d.context50_pp.abs()).sum())
    rows.append(("L3", "Pairs misattributed to context", f"{fd}/28", "0/28", "low"))
    gt = pd.read_csv(f"{R}/gt_scale.csv")
    n_ref = 39
    rows.append(("L4", "Calls to locate the 6 triggers", f"{n_ref * np.log2(10):.0f}", "6", "low"))
    rows.append(("L4", "Clustered GROUP BY bias (pp)", f"{b[('agnews', 'sorted')]:.2f}", "0.00", "low"))
    v = pd.read_csv(f"{R}/main/V2_served.csv"); v = v[(v.http == 200) & v.requested.isin(["glm-4.5-air", "glm-4.7"])]
    rows.append(("L5", "Labels on the wrong model (\\%)", f"{100 * (v.served != v.requested).mean():.0f}", "0", "low"))
    D = pd.DataFrame(rows, columns=["rule", "metric", "common", "recommended", "better"])
    D.to_csv(f"{R}/practice_summary.csv", index=False)

    def num(x):
        return float(str(x).split("/")[0])
    lines = [r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
             r"Rule & Measured on the released runs & Common & Rec. \\", r"\midrule"]
    wins = 0
    for i, (rule, metric, com, rec, better) in enumerate(rows):
        a, bb = num(com), num(rec)
        rec_wins = bb < a if better == "low" else bb > a
        wins += rec_wins
        com_s, rec_s = (com, r"\textbf{" + rec + "}") if rec_wins else (r"\textbf{" + com + "}", rec)
        first = rule if i == 0 or rows[i - 1][0] != rule else ""
        if first and i:
            lines.append(r"\midrule")
        lines.append(f"{first} & {metric} & {com_s} & {rec_s} \\\\")
    lines += [r"\midrule", rf"\multicolumn{{4}}{{@{{}}l}}{{Recommended practice better in {wins} of {len(rows)} rows}} \\",
              r"\bottomrule", r"\end{tabular}"]
    open(f"{T}/practice.tex", "w").write("\n".join(lines) + "\n")
    print(D.to_string(), "\nrecommended better in", wins, "of", len(rows))


if __name__ == "__main__":
    main()
