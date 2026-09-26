import io, json
import numpy as np, pandas as pd
from ubiomark import cli, novelty


def test_novelty_query_and_count_with_fake_opener():
    q = novelty.build_query("TP53", "fibromyalgia")
    assert q == '"TP53"[tiab] AND fibromyalgia[tiab]'
    fake = lambda url, timeout=60: io.StringIO(json.dumps({"esearchresult": {"count": "7"}}))
    assert novelty.pubmed_count(q, sleep=0, opener=fake) == 7


def test_cli_meta_roundtrip(tmp_path):
    for i, g in enumerate([0.4, 0.6]):
        pd.DataFrame({"gene": ["A", "B"], "g": [g, 0.0], "v": [0.1, 0.1]}).to_csv(tmp_path / f"s{i}.csv", index=False)
    out = tmp_path / "m.csv"
    cli.main(["meta", str(tmp_path / "s0.csv"), str(tmp_path / "s1.csv"), "--out", str(out)])
    m = pd.read_csv(out, index_col=0)
    assert np.isclose(m.loc["A", "mu"], 0.5) and m.loc["A", "k"] == 2
