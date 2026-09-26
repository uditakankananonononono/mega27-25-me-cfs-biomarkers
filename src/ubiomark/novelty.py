"""Literature novelty check: PubMed title/abstract co-mention counts via NCBI E-utilities."""
from __future__ import annotations
import json, time, urllib.parse, urllib.request

DISEASE_QUERY = {
    "postpartum_depression": '("postpartum depression"[tiab] OR "postnatal depression"[tiab] OR "perinatal depression"[tiab])',
    "me_cfs": '("chronic fatigue syndrome"[tiab] OR "myalgic encephalomyelitis"[tiab])',
    "fibromyalgia": 'fibromyalgia[tiab]',
    "endometriosis": 'endometriosis[tiab]',
    "preeclampsia": '(preeclampsia[tiab] OR pre-eclampsia[tiab])',
    "pcos": '("polycystic ovary"[tiab] OR PCOS[tiab])',
    "long_covid": '("long covid"[tiab] OR "post-acute sequelae"[tiab] OR "post-COVID"[tiab])',
    "chagas": '(Chagas[tiab] OR "Trypanosoma cruzi"[tiab])',
    "leishmaniasis": 'leishmaniasis[tiab]',
    "interstitial_cystitis": '("interstitial cystitis"[tiab] OR "bladder pain syndrome"[tiab])',
}


def build_query(gene: str, disease: str) -> str:
    return f'"{gene}"[tiab] AND {DISEASE_QUERY[disease]}'


def pubmed_count(term: str, sleep: float = 0.35, opener=urllib.request.urlopen) -> int:
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=0&term="
           + urllib.parse.quote(term))
    for i in range(4):
        try:
            n = int(json.load(opener(url, timeout=60))["esearchresult"]["count"])
            time.sleep(sleep)
            return n
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(term)


def comention(gene: str, disease: str, **kw) -> int:
    return pubmed_count(build_query(gene, disease), **kw)
