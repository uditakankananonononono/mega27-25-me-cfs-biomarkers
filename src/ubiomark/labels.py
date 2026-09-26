"""Rule-based case/control labelling of GEO samples from their annotation.

Rules are explicit and auditable; ambiguous samples are dropped, never guessed.
Only 'grouping' characteristics (diagnosis/disease/condition/status/group...) plus title/source are read,
so nuisance fields such as 'migraine: no' cannot flip a label."""
from __future__ import annotations
import re
import pandas as pd

TERMS = {
    "postpartum_depression": r"postpartum depress\w*|post-partum depress\w*|\bppd\b",
    "me_cfs": r"chronic fatigue|\bcfs\b|me/cfs|myalgic|\bme\b",
    "fibromyalgia": r"fibromyalg\w*|\bfm\b|\bfms\b",
    "endometriosis": r"endometrio(sis|ma|tic)|ectopic|\blesions?\b|\bendo\b|peritoneal implant",
    "preeclampsia": r"pre-?eclamp\w*|\bpe\b|\bpet\b|hellp|\bpre\b|\bspe\b|\bepe\b|\blope\b",
    "pcos": r"polycystic|\bpcos\b",
    "long_covid": r"long[- ]?covid|post-?covid|\bpasc\b|post-acute",
    "chagas": r"chagas\w*|\bccc\b|t\.? ?cruzi|trypanosoma|chagasic",
    "leishmaniasis": r"leishman\w*|\bcl\b|\bvl\b|\blcl\b|\bmcl\b|kala|\bpkdl\b",
    "interstitial_cystitis": r"interstitial cystitis|\bic\b|bladder pain|\bbps\b|hunner|ic/bps",
}
GENERIC_CASE = r"^(case|cases|patient|patients|disease|affected)$"
GENERIC_CONTROL = (r"\b(healthy|controls?|normal|unaffected|uninfected|non[- ]?infected|mock|ctrl|hc|nc|"
                   r"euthymic|normotensive|disease[- ]free|normal pregnancy|uncomplicated|term control)\b")
GROUP_KEYS = r"diagnos|disease|condition|status|group|state|phenotype|class|cohort|subject|patient|case|clinical|infection|pathology"
EXCLUDE = r"\b(treated with|stimulat|knock|sirna|shrna|transfect|overexpress|cell line|culture|in vitro)\w*"

# explicit, audited per-series overrides: restrict to a subset and/or redefine case/control
OVERRIDES = {
    # Mehta/Osborne cohort: longitudinal; use the postpartum timepoint only; drop 'always depressed'
    ("postpartum_depression", "GSE45603"): {"subset": r"time: postpartum", "case": r"condition: ppd",
                                             "control": r"condition: (euthymic|control)"},
}


def _fields(ann: pd.DataFrame) -> pd.DataFrame:
    return ann[[c for c in ann.columns if c.startswith(("Sample_title", "Sample_source_name", "Sample_characteristics"))]].astype(str)


def sample_text(ann: pd.DataFrame) -> pd.Series:
    return _fields(ann).agg(" | ".join, axis=1).str.lower()


def group_text(ann: pd.DataFrame) -> pd.Series:
    f = _fields(ann)
    out = []
    for _, row in f.iterrows():
        parts = []
        for c, v in row.items():
            v = v.lower().strip()
            if c.startswith("Sample_characteristics"):
                k = v.split(":", 1)[0] if ":" in v else ""
                if re.search(GROUP_KEYS, k) or re.search(r"\b(ppd|pcos)\b", k):
                    parts.append(v)
            else:
                parts.append(v)
        out.append(" | ".join(parts))
    return pd.Series(out, index=ann.index)


YES = r"(y|yes|1|true|positive|pos|affected)"
NO = r"(n|no|0|false|negative|neg|unaffected)"


def label_one(text: str, disease: str) -> str | None:
    text = text.replace("_", " ")
    if re.search(EXCLUDE, text):
        return None
    term = TERMS[disease]
    # explicit indicator fields, e.g. "pcos: n" / "ppd: yes"
    indicator_votes = set()
    for part in text.split("|"):
        if ":" in part:
            k, v = [x.strip() for x in part.split(":", 1)]
            if re.search(term, k):
                if re.fullmatch(YES, v):
                    indicator_votes.add("case")
                if re.fullmatch(NO, v):
                    indicator_votes.add("control")
    if len(indicator_votes) > 1:
        return None  # contradictory disease indicators must not be selected by field order
    if indicator_votes:
        return next(iter(indicator_votes))
    neg = rf"\b(non[- ]?|no |without |not |negative for )({term})"
    t2 = re.sub(neg, " __ctrl__ ", text)
    ctrl = bool(re.search(GENERIC_CONTROL, t2)) or "__ctrl__" in t2
    fields = [p.split(":", 1)[-1].strip() for p in t2.split("|")]
    case = bool(re.search(term, t2)) or any(re.fullmatch(GENERIC_CASE, f) for f in fields)
    if ctrl and not case:
        return "control"
    if case and not ctrl:
        return "case"
    return None


def label_samples(ann: pd.DataFrame, disease: str, gse: str | None = None) -> pd.Series:
    full = sample_text(ann)
    ov = OVERRIDES.get((disease, gse))
    if ov:
        def f(t):
            if ov.get("subset") and not re.search(ov["subset"], t):
                return None
            if re.search(ov["case"], t):
                return "case"
            if re.search(ov["control"], t):
                return "control"
            return None
        return full.map(f)
    return group_text(ann).map(lambda t: label_one(t, disease))
