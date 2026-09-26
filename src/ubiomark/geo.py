"""GEO series-matrix and platform download + parsing."""
from __future__ import annotations
import gzip, io, os, re, time, urllib.request
import numpy as np
import pandas as pd

CACHE = os.environ.get("UBIOMARK_CACHE", os.path.expanduser("~/mega27-25/data/raw"))


def _fetch(url: str, dest: str, tries: int = 4) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest + ".part", "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(dest + ".part", dest)
            return dest
        except Exception as e:  # network retry
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")


def series_dir(gse: str) -> str:
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}/matrix/"


def list_matrix_files(gse: str) -> list[str]:
    html = urllib.request.urlopen(series_dir(gse), timeout=60).read().decode()
    return sorted(set(re.findall(r'href="(GSE\d+(?:-GPL\d+)?_series_matrix\.txt\.gz)"', html)))


def download_matrices(gse: str) -> list[str]:
    return [_fetch(series_dir(gse) + f, os.path.join(CACHE, "matrix", f)) for f in list_matrix_files(gse)]


def parse_series_matrix(path_or_text) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (expression probes x samples, sample annotation df, series meta)."""
    if isinstance(path_or_text, str) and os.path.exists(path_or_text):
        return _parse_matrix_file(path_or_text)
    text = path_or_text
    meta: dict = {}
    sample_rows: dict[str, list[list[str]]] = {}
    lines = text.splitlines()
    try:
        start = lines.index("!series_matrix_table_begin")
        end = lines.index("!series_matrix_table_end")
    except ValueError:
        start = end = len(lines)
    for ln in lines[:start]:
        if not ln.startswith("!"):
            continue
        parts = ln.split("\t")
        key = parts[0][1:]
        vals = [p.strip().strip('"') for p in parts[1:]]
        if key.startswith("Sample_"):
            sample_rows.setdefault(key, []).append(vals)
        else:
            meta.setdefault(key, []).append(" ".join(vals))
    table = "\n".join(lines[start + 1:end])
    if table.strip():
        expr = pd.read_csv(io.StringIO(table), sep="\t", index_col=0, quotechar='"', low_memory=False)
        expr = expr.apply(pd.to_numeric, errors="coerce")
    else:
        expr = pd.DataFrame()
    ids = sample_rows.get("Sample_geo_accession", [[]])[0]
    ann = pd.DataFrame(index=ids)
    for key, rows in sample_rows.items():
        for j, vals in enumerate(rows):
            if len(vals) == len(ids):
                ann[f"{key}_{j}" if len(rows) > 1 else key] = vals
    return expr, ann, meta


def _parse_matrix_file(path: str):
    """Memory-lean parser: header lines read as text, table streamed by the C CSV engine as float32."""
    opener = gzip.open if path.endswith(".gz") else open
    head = []
    with opener(path, "rt", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("!series_matrix_table_begin"):
                break
            head.append(ln.rstrip("\n"))
    _, ann, meta = parse_series_matrix("\n".join(head))
    # Read the matrix table from its physical boundary. pandas skiprows counts CSV
    # records, not physical lines, when a quoted GEO metadata field spans lines;
    # passing the original file and a physical line count can silently skip the
    # actual ID_REF/GSM header (as in GSE621).
    with opener(path, "rt", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("!series_matrix_table_begin"):
                break
        else:
            raise ValueError(f"GEO series matrix table start not found: {path}")
        header = next(fh).rstrip("\r\n")
        ids = header.split("\t")[1:]
        ids = [v.strip().strip('"') for v in ids]
        if ids != list(ann.index):
            raise ValueError(f"GEO matrix header/annotation GSM mismatch: {path}")
        # Stream only table rows into a temporary file so pandas sees the exact
        # header and no multiline metadata or trailing GEO marker.
        import tempfile
        with tempfile.TemporaryFile(mode="w+t") as matrix:
            matrix.write(header + "\n")
            for ln in fh:
                if ln.startswith("!series_matrix_table_end"):
                    break
                matrix.write(ln)
            else:
                raise ValueError(f"GEO series matrix table end not found: {path}")
            matrix.seek(0)
            expr = pd.read_csv(matrix, sep="\t", index_col=0, quotechar='"',
                               low_memory=False, na_values=["null", "NA", ""])
    if list(expr.columns) != list(ann.index):
        raise ValueError(f"Parsed GEO matrix header/annotation GSM mismatch: {path}")
    expr = expr.apply(pd.to_numeric, errors="coerce").astype("float32")
    return expr, ann, meta


SYMBOL_COLS = ["Gene Symbol", "GENE SYMBOL", "GENE_SYMBOL", "Symbol", "ILMN_Gene", "gene_symbol", "GeneSymbol",
               "Gene symbol", "SYMBOL", "ORF", "gene_assignment", "GENE_NAME", "Gene_Symbol"]


def _gpl_ftp(gpl: str) -> str:
    stem = gpl[:-3] + "nnn" if len(gpl) > 6 else "GPLnnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{stem}/{gpl}/"


def platform_table(gpl: str) -> pd.DataFrame:
    """Prefer the compact FTP .annot.gz (GEO-curated 'Gene symbol'); fall back to the full acc.cgi table."""
    try:
        dest = _fetch(_gpl_ftp(gpl) + f"annot/{gpl}.annot.gz", os.path.join(CACHE, "gpl", f"{gpl}.annot.gz"), tries=2)
        with gzip.open(dest, "rt", errors="replace") as fh:
            lines = [l for l in fh if not l.startswith(("^", "!", "#"))]
        return pd.read_csv(io.StringIO("".join(lines)), sep="\t", dtype=str, low_memory=False)
    except Exception:
        pass
    dest = os.path.join(CACHE, "gpl", f"{gpl}.txt")
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl}&targ=self&form=text&view=data"
    _fetch(url, dest)
    with open(dest, errors="replace") as fh:
        lines = [l for l in fh if not l.startswith(("^", "!", "#"))]
    return pd.read_csv(io.StringIO("".join(lines)), sep="\t", dtype=str, low_memory=False)


def _clean_symbol(s: str, col: str) -> str | None:
    if not isinstance(s, str) or not s.strip() or s.strip() in ("---", "NA", "nan"):
        return None
    if col == "gene_assignment":  # "NM_x // SYMBOL // desc // ..."
        parts = [p.strip() for p in s.split("//")]
        s = parts[1] if len(parts) > 1 else ""
    toks = [t.strip() for t in re.split(r"\s*///\s*|\s*;\s*|,", s.strip()) if t.strip()]
    good = [t for t in toks if not re.match(r"(MIR\d|LOC\d|SNOR|LINC)", t)]
    s = (good or toks or [""])[0]
    return s if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", s or "") else None


def probe_to_symbol(gpl: str) -> pd.Series:
    t = platform_table(gpl)
    col = next((c for c in SYMBOL_COLS if c in t.columns), None)
    if col is None:
        hg = hgnc_maps()
        for c, key, f in [("ENTREZ_GENE_ID", "entrez", lambda v: str(v).split(".")[0]),
                          ("GB_ACC", "refseq", lambda v: str(v).split(".")[0])]:
            if c in t.columns:
                m = t.set_index(t.columns[0])[c].map(lambda v: hg[key].get(f(v))).dropna()
                if len(m) > 1000:
                    m.index = m.index.astype(str)
                    return m
        raise KeyError(f"{gpl}: no symbol column in {list(t.columns)[:15]}")
    m = t.set_index(t.columns[0])[col].map(lambda s: _clean_symbol(s, col)).dropna()
    m.index = m.index.astype(str)
    return m


HGNC_PATH = os.path.join(CACHE, "hgnc", "hgnc_complete_set.txt")
_HG = None


def hgnc_maps() -> dict:
    """Entrez ID / RefSeq accession -> approved HGNC symbol (HGNC complete set)."""
    global _HG
    if _HG is None:
        _fetch("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt", HGNC_PATH)
        h = pd.read_csv(HGNC_PATH, sep="\t", dtype=str, usecols=["symbol", "entrez_id", "refseq_accession"])
        ent = dict(zip(h.entrez_id.dropna(), h.symbol[h.entrez_id.notna()]))
        ref = {}
        for sym, rs in zip(h.symbol, h.refseq_accession.fillna("")):
            for r in rs.split("|"):
                if r:
                    ref[r.split(".")[0]] = sym
        _HG = {"entrez": ent, "refseq": ref}
    return _HG


def to_gene_level(expr: pd.DataFrame, p2s: pd.Series) -> pd.DataFrame:
    """Collapse probes to genes by the probe with the highest mean (standard maxMean rule), log2 if needed."""
    x = expr.copy()
    x.index = x.index.astype(str)
    x = x.loc[x.index.intersection(p2s.index)]
    x = log2_if_needed(x)
    x["__sym"] = p2s.loc[x.index].values
    x["__mean"] = x.drop(columns="__sym").mean(axis=1)
    x = x.sort_values("__mean", ascending=False).drop_duplicates("__sym")
    return x.set_index("__sym").drop(columns="__mean")


def log2_if_needed(x: pd.DataFrame) -> pd.DataFrame:
    v = x.to_numpy(dtype=float)
    q = np.nanquantile(v, [0.99, 0.25]) if np.isfinite(v).any() else [0, 0]
    if q[0] > 100:  # clearly linear scale (GEO2R rule)
        v = np.where(v <= 0, np.nan, v)
        return pd.DataFrame(np.log2(v), index=x.index, columns=x.columns)
    return x
