import pandas as pd
from ubiomark import labels, geo

MATRIX = """!Series_title\t"Toy"
!Sample_title\t"C1"\t"F1"\t"F2"\t"X"
!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"\t"GSM4"
!Sample_platform_id\t"GPL1"\t"GPL1"\t"GPL1"\t"GPL1"
!Sample_characteristics_ch1\t"diagnosis: healthy control"\t"diagnosis: fibromyalgia"\t"diagnosis: fibromyalgia"\t"diagnosis: unknown"
!Sample_characteristics_ch1\t"migraine: no"\t"migraine: yes"\t"migraine: no"\t"migraine: no"
!series_matrix_table_begin
"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"\t"GSM4"
"p1"\t1\t2\t3\t4
"p2"\t5\t6\t7\t8
!series_matrix_table_end
"""


def test_parse_series_matrix():
    e, a, m = geo.parse_series_matrix(MATRIX)
    assert e.shape == (2, 4) and list(a.index) == ["GSM1", "GSM2", "GSM3", "GSM4"]
    assert a["Sample_platform_id"].iloc[0] == "GPL1"


def test_nuisance_field_does_not_flip_label():
    _, a, _ = geo.parse_series_matrix(MATRIX)
    lab = labels.label_samples(a, "fibromyalgia")
    assert lab.tolist() == ["control", "case", "case", None]


def test_wrong_disease_yields_no_cases():
    _, a, _ = geo.parse_series_matrix(MATRIX)
    assert (labels.label_samples(a, "me_cfs") == "case").sum() == 0


def test_negation_is_control():
    assert labels.label_one("status: non-preeclamptic", "preeclampsia") == "control"
    assert labels.label_one("status: preeclampsia", "preeclampsia") == "case"
    assert labels.label_one("treated with lps | preeclampsia", "preeclampsia") is None


def test_gene_level_collapse_max_mean():
    e = pd.DataFrame({"s1": [1.0, 3.0, 2.0], "s2": [1.0, 3.0, 2.0]}, index=["a", "b", "c"])
    p2s = pd.Series({"a": "G1", "b": "G1", "c": "G2"})
    g = geo.to_gene_level(e, p2s)
    assert g.loc["G1", "s1"] == 3.0 and g.loc["G2", "s1"] == 2.0


def test_clean_symbol_prefers_protein_coding():
    assert geo._clean_symbol("MIR4640 /// DDR1", "Gene symbol") == "DDR1"
    assert geo._clean_symbol("NM_1 // TP53 // tumor protein", "gene_assignment") == "TP53"
    assert geo._clean_symbol("---", "Gene symbol") is None


def test_indicator_fields_and_underscores():
    assert labels.label_one("liver | pcos: n | cardiovascular disease: y", "pcos") == "control"
    assert labels.label_one("sample | ppd: yes", "postpartum_depression") == "case"
    assert labels.label_one("ep1_adipose_control | omental adipose tissue", "pcos") == "control"
    assert labels.label_one("ep1_adipose_pcos | omental adipose tissue", "pcos") == "case"


def test_conflicting_disease_indicators_abstain_in_either_order():
    for text in ("pcos: n | pcos: y", "pcos: yes | pcos: no"):
        assert labels.label_one(text, "pcos") is None
    assert labels.label_one("ppd: yes | ppd: yes", "postpartum_depression") == "case"


def test_file_parser_multiline_quoted_summary_preserves_exact_gsm_header(tmp_path):
    import gzip
    source = MATRIX.replace('!Series_title\t"Toy"', '!Series_title\t"Toy"\n!Series_summary\t"A multiline summary\ncontinued on next line"')
    path = tmp_path / 'toy_series_matrix.txt.gz'
    with gzip.open(path, 'wt') as f:
        f.write(source)
    e, a, _ = geo.parse_series_matrix(str(path))
    assert e.shape == (2, 4)
    assert list(e.columns) == list(a.index) == ['GSM1', 'GSM2', 'GSM3', 'GSM4']
    assert e.loc['p2', 'GSM4'] == 8
