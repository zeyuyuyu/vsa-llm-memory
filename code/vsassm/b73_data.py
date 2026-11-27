"""B73 (prereg/B73_confirmatory_prereg.md): the FRESH equal-bit entity set (Wikidata, outside PopQA).

python3 b73_data.py fetch OUTDIR        # one SPARQL query per class (CLASSES) -> OUTDIR/raw/{cls}.json.gz + fetch_log.json
python3 b73_data.py build OUTDIR        # raw -> OUTDIR/b32_data/entities.json, emb_strings.json (frozen rule below)
python3 b73_data.py leak OUTDIR         # leakage check against PopQA's entity file (and the raw PopQA subjects)
python3 b73_data.py stats OUTDIR
python3 b73_data.py synth OUTDIR        # SYNTHETIC entity set + random codes for the self-test (no Wikidata)

Selection rule (frozen in the pre-registration before any fetch):
  1. For each class C in CLASSES (PopQA's subject kinds: people, films, literary works, albums, TV series, singles,
     cities, football clubs, businesses), one query to https://query.wikidata.org/sparql: a random sample of
     SAMPLE[C] items with wdt:P31 C (Blazegraph's bd:sample service, sampleType RANDOM), kept if the item has an
     English Wikipedia article (as PopQA's subjects: PopQA's popularity is English Wikipedia page views), an English
     label and at least one English alias (skos:altLabel). Rows: item, label, alias. Raw responses saved with the fetch
     time. A query that fails or times out is re-sent once with half the sample size (recorded); read-only.
  2. Items: label = the English label; aliases = the English aliases, sorted (the response order is not defined).
     An item that appears under two classes is kept under the first class in CLASSES order.
  3. Leakage filter (PopQA = b49_out/b32_data/entities.json and the raw subjects of b32_data/raw/popqa_test.tsv):
     drop the item if its URI is a PopQA subject or its normalized label equals the normalized form of any PopQA
     label or alias; drop an alias whose normalized form equals the normalized form of any PopQA label or alias, is
     empty, or equals the item's normalized label; drop the item if no alias is left.
  4. Per class: order the survivors by sha256('b73order|' + uri) and keep the first QUOTA[C].
  5. B32's filters (b32_data.build, unchanged logic): drop entities whose normalized label is shared with another
     entity; drop an alias whose normalized form is the label or alias of a different entity, or repeats one of the
     entity's own aliases. Entities left without an alias stay (keys without queries, as in PopQA).
  6. Split by b32_data.split_of(uri) (sha256('b32|' + uri): train 30 %, dev 20 %, test 50 %).
"""
import collections
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b32_data as D  # noqa: E402  (norm, split_of)

ENDPOINT = 'https://query.wikidata.org/sparql'
UA = 'research-script/0.1 (generic)'
# (name, QID, sample size, quota)
CLASSES = [('human', 'Q5', 40000, 2600), ('film', 'Q11424', 25000, 2000), ('literary_work', 'Q7725634', 40000, 800),
           ('album', 'Q482994', 30000, 700), ('tv_series', 'Q5398426', 15000, 600), ('single', 'Q134556', 15000, 400),
           ('city', 'Q515', 15000, 700), ('football_club', 'Q476028', 20000, 600), ('business', 'Q4830453', 30000, 600)]
POPQA_ENT = os.path.join(ROOT, 'b49_out', 'b32_data', 'entities.json')
POPQA_RAW = os.path.join(ROOT, 'b32_data', 'raw', 'popqa_test.tsv')

QUERY = """SELECT ?item ?label ?alias WHERE {
  SERVICE bd:sample { ?item wdt:P31 wd:%s . bd:serviceParam bd:sample.limit %d . bd:serviceParam bd:sample.sampleType "RANDOM" . }
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
  ?item rdfs:label ?label . FILTER(LANG(?label) = "en")
  ?item skos:altLabel ?alias . FILTER(LANG(?alias) = "en")
}"""


def sparql(q, timeout=150):
    """curl, as the count-only probe (compute note 1: python-urllib's requests were refused with 403 / 429)."""
    import subprocess
    r = subprocess.run(['curl', '-s', '-f', '-m', str(timeout), '-G', ENDPOINT, '--data-urlencode', f'query={q}',
                        '-H', 'Accept: application/sparql-results+json', '-A', UA], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f'curl rc={r.returncode} {r.stderr[:200]!r}')
    return r.stdout


def fetch(outdir):
    os.makedirs(os.path.join(outdir, 'raw'), exist_ok=True)
    log = []
    for name, qid, n, _ in CLASSES:
        fn = os.path.join(outdir, 'raw', f'{name}.json.gz')
        if os.path.exists(fn):
            continue
        for attempt, ns in enumerate((n, n // 2)):
            q = QUERY % (qid, ns)
            t0 = time.time()
            stamp = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            try:
                body = sparql(q)
                json.loads(body)
            except Exception as e:  # noqa: BLE001
                log.append({'class': name, 'qid': qid, 'sample': ns, 'attempt': attempt, 'start': stamp,
                            'sec': round(time.time() - t0, 1), 'error': repr(e)[:300], 'query': q})
                print('fail', name, ns, repr(e)[:120], flush=True)
                time.sleep(30)
                continue
            with gzip.open(fn, 'wb') as fo:
                fo.write(body)
            nrows = len(json.loads(body)['results']['bindings'])
            log.append({'class': name, 'qid': qid, 'sample': ns, 'attempt': attempt, 'start': stamp,
                        'sec': round(time.time() - t0, 1), 'rows': nrows, 'bytes': len(body), 'query': q,
                        'md5': hashlib.md5(body).hexdigest()})
            print('ok', name, ns, nrows, 'rows', f'{time.time() - t0:.0f}s', flush=True)
            break
        time.sleep(15)  # one query at a time, paced (compute note 1)
        json.dump(log, open(os.path.join(outdir, 'fetch_log.json'), 'w'), indent=1)


def popqa_strings():
    d = json.load(open(POPQA_ENT))
    uris, strs = set(), set()
    for e in d['entities']:
        uris.add(e['uri'])
        strs.add(D.norm(e['name']))
        strs.update(D.norm(a) for a in e['aliases'])
    raw_uris = set()
    if os.path.exists(POPQA_RAW):
        import csv
        for r in csv.DictReader(open(POPQA_RAW, encoding='utf-8'), delimiter='\t'):
            raw_uris.add(r['s_uri'])
            strs.add(D.norm(r['subj']))
            strs.update(D.norm(a) for a in json.loads(r['s_aliases']))
    strs.discard('')
    return uris | raw_uris, strs, {e['name'] for e in d['entities']} | {a for e in d['entities'] for a in e['aliases']}


def build(outdir):
    pq_uris, pq_norm, _ = popqa_strings()
    items, order = {}, []
    n_rows = {}
    for name, qid, _, quota in CLASSES:
        fn = os.path.join(outdir, 'raw', f'{name}.json.gz')
        b = json.loads(gzip.open(fn).read())['results']['bindings']
        n_rows[name] = len(b)
        for r in b:
            u = r['item']['value']
            if u in items and items[u]['cls'] != name:
                continue
            e = items.setdefault(u, {'uri': u, 'name': r['label']['value'], 'aliases': set(), 'cls': name})
            if r['label']['value'] != e['name']:
                raise SystemExit(f'two English labels for {u}')
            e['aliases'].add(r['alias']['value'].strip())
    st = collections.Counter()
    kept = []
    for name, qid, _, quota in CLASSES:
        cand = []
        for u, e in items.items():
            if e['cls'] != name:
                continue
            st[name, 'items'] += 1
            nl = D.norm(e['name'])
            if u in pq_uris or not nl or nl in pq_norm:
                st[name, 'drop_popqa_or_label'] += 1
                continue
            al = sorted(a for a in e['aliases'] if a and D.norm(a) and D.norm(a) != nl and D.norm(a) not in pq_norm)
            if not al:
                st[name, 'drop_no_alias_left'] += 1
                continue
            cand.append({'uri': u, 'name': e['name'], 'aliases': al, 'props': [name]})
        cand.sort(key=lambda x: hashlib.sha256(('b73order|' + x['uri']).encode()).hexdigest())
        st[name, 'candidates'] = len(cand)
        kept += cand[:quota]
        st[name, 'kept'] = min(quota, len(cand))
    # B32's filters (b32_data.build's logic)
    ents = {e['uri']: e for e in kept}
    n_raw = len(ents)
    lab = collections.Counter(D.norm(e['name']) for e in ents.values())
    ents = {u: e for u, e in ents.items() if lab[D.norm(e['name'])] == 1}
    n_dup = n_raw - len(ents)
    owner = collections.defaultdict(set)
    for u, e in ents.items():
        owner[D.norm(e['name'])].add(u)
        for a in e['aliases']:
            owner[D.norm(a)].add(u)
    n_al0 = sum(len(e['aliases']) for e in ents.values())
    for u, e in ents.items():
        keep, seen = [], set()
        for a in e['aliases']:
            na = D.norm(a)
            if owner[na] != {u} or na in seen:
                continue
            seen.add(na)
            keep.append(a)
        e['aliases'] = keep
    n_al1 = sum(len(e['aliases']) for e in ents.values())
    out = []
    for u in sorted(ents):
        e = ents[u]
        e['split'] = D.split_of(u)
        out.append(e)
    bd = os.path.join(outdir, 'b32_data')
    os.makedirs(bd, exist_ok=True)
    json.dump({'source': 'Wikidata SPARQL (b73_data.py fetch; raw/ + fetch_log.json)', 'n_rows': n_rows,
               'class_stats': {f'{a}|{b}': v for (a, b), v in st.items()}, 'n_raw_entities': n_raw,
               'n_dropped_duplicate_label': n_dup, 'n_aliases_before_ambiguity_filter': n_al0, 'n_aliases': n_al1,
               'entities': out}, open(os.path.join(bd, 'entities.json'), 'w'), ensure_ascii=False, indent=0)
    S, seen = [], set()
    for e in out:
        for s in [e['name']] + e['aliases']:
            if s not in seen:
                seen.add(s)
                S.append(s)
    json.dump(S, open(os.path.join(bd, 'emb_strings.json'), 'w'), ensure_ascii=False)
    print(f'{n_raw} entities, {n_dup} dropped (shared label), aliases {n_al0} -> {n_al1}; {len(S)} strings')
    stats(outdir)


def leak(outdir):
    """Leakage check: no fresh entity URI, label or alias string (exact or normalized) in PopQA's entity file (or the
    raw PopQA subjects)."""
    pq_uris, pq_norm, pq_exact = popqa_strings()
    d = json.load(open(os.path.join(outdir, 'b32_data', 'entities.json')))
    bad = collections.Counter()
    n_s = 0
    for e in d['entities']:
        bad['uri'] += e['uri'] in pq_uris
        for s in [e['name']] + e['aliases']:
            n_s += 1
            bad['exact'] += s in pq_exact
            bad['norm'] += D.norm(s) in pq_norm
    res = {'n_entities': len(d['entities']), 'n_strings_checked': n_s, 'uri_in_popqa': bad['uri'],
           'string_exact_in_popqa': bad['exact'], 'string_normalized_in_popqa': bad['norm'],
           'popqa_uris': len(pq_uris), 'popqa_normalized_strings': len(pq_norm)}
    print(json.dumps(res))
    return res


def stats(outdir):
    d = json.load(open(os.path.join(outdir, 'b32_data', 'entities.json')))
    c = collections.Counter()
    for e in d['entities']:
        c[e['split'], 'ent'] += 1
        c[e['split'], 'ent_alias'] += bool(e['aliases'])
        c[e['split'], 'alias'] += len(e['aliases'])
        c[e['props'][0], e['split']] += 1
    for s in ('train', 'dev', 'test'):
        print(f'{s:5s} entities {c[s, "ent"]:5d}  with alias {c[s, "ent_alias"]:5d}  aliases {c[s, "alias"]:5d}')
    print({name: [c[name, s] for s in ('train', 'dev', 'test')] for name, *_ in CLASSES})


DIMS = {'qwen8_mean_L9_f16': 4096, 'b46_granite33i_mean_L30_f16': 4096, 'llama31i_mean_L24_f16': 4096,
        'mistral7i_eol_L24_f16': 4096, 'e5_base': 768, 'bge_small': 384, 'minilm': 384}


def layout(root):
    """The directory layout every module's loader expects (b53_run.enc_dir, b60_run / b64_run DATA = root/b49_out)."""
    bd = os.path.join(root, 'b32_data')
    for sub in ('b46_out', 'b49_out', 'b49_out/enc_llama', 'b49_out/enc_mistral', 'b49_out/enc_bge',
                'b49_out/enc_minilm'):
        d = os.path.join(root, sub)
        os.makedirs(d, exist_ok=True)
        ln = os.path.join(d, 'b32_data')
        if not os.path.islink(ln):
            os.symlink(os.path.relpath(bd, d), ln)


def synth(outdir, n_ent=3000, seed=73):
    """Synthetic self-test data: random names, 0-3 aliases each, codes = label direction + noise (aliases near their
    label), the paper's code dimensions. No Wikidata, no PopQA."""
    import numpy as np
    rng = np.random.default_rng(seed)
    syl = ['ka', 'lo', 'mi', 'ru', 'te', 'sa', 'no', 'vi', 'de', 'po', 'zu', 'xi']
    ents = []
    for i in range(n_ent):
        name = ''.join(rng.choice(syl, 4)) + f' {i}'
        al = [f'{name} alias{j}' for j in range(int(rng.integers(0, 4)))]
        u = f'synthetic:{i}'
        ents.append({'uri': u, 'name': name, 'aliases': al, 'props': ['synthetic'], 'split': D.split_of(u)})
    bd = os.path.join(outdir, 'b32_data')
    os.makedirs(bd, exist_ok=True)
    json.dump({'source': 'synthetic', 'entities': ents}, open(os.path.join(bd, 'entities.json'), 'w'))
    S = [s for e in ents for s in [e['name']] + e['aliases']]
    json.dump(S, open(os.path.join(bd, 'emb_strings.json'), 'w'))
    owner = {s: k for k, e in enumerate(ents) for s in [e['name']] + e['aliases']}
    for tag, n in DIMS.items():
        base = rng.standard_normal((n_ent, n)).astype(np.float32)
        X = np.stack([base[owner[s]] + 0.9 * rng.standard_normal(n).astype(np.float32) for s in S])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        np.save(os.path.join(bd, f'emb_{tag}.npy'), X.astype(np.float16) if tag.endswith('_f16') else X)
    layout(outdir)
    print('synthetic', n_ent, 'entities', len(S), 'strings ->', outdir)


if __name__ == '__main__':
    cmd, od = sys.argv[1], sys.argv[2]
    {'fetch': fetch, 'build': build, 'leak': leak, 'stats': stats, 'synth': synth, 'layout': layout}[cmd](od)
