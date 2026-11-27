"""B32 data (prereg/B32_*.md): an entity pool with real Wikidata aliases (PopQA subjects), split by entity.

python3 b32_data.py build            # b32_data/raw/popqa_test.tsv -> b32_data/entities.json
python3 b32_data.py stats

Every PopQA subject (one per Wikidata URI) is an entity; its key string is PopQA's subject label. An alias is any of
PopQA's s_aliases (from Wikidata) whose normalized form differs from the entity's own label. Entities whose normalized
label is shared with another entity are dropped (a store could not tell them apart). An alias whose normalized form
equals the label or another alias of a different entity is dropped (its gold answer would be ambiguous). Entities are
split by a hash of the URI: train 30%, dev 20%, test 50%. Facts are not taken from PopQA: each store assigns random
values (b32_common.make_store), so no value can be guessed from a name.
"""
import csv
import collections
import hashlib
import json
import re
import sys

RAW = 'b32_data/raw/popqa_test.tsv'
OUT = 'b32_data/entities.json'


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def split_of(uri):
    h = int(hashlib.sha256(('b32|' + uri).encode()).hexdigest()[:8], 16) % 100
    return 'train' if h < 30 else ('dev' if h < 50 else 'test')


def build():
    rows = list(csv.DictReader(open(RAW, encoding='utf-8'), delimiter='\t'))
    ents = {}
    for r in rows:
        e = ents.setdefault(r['s_uri'], {'uri': r['s_uri'], 'name': r['subj'], 'aliases': [], 'props': []})
        if r['subj'] != e['name']:
            raise SystemExit(f'two labels for {r["s_uri"]}: {e["name"]!r} {r["subj"]!r}')
        if r['prop'] not in e['props']:
            e['props'].append(r['prop'])
        for a in json.loads(r['s_aliases']):
            a = a.strip()
            if a and norm(a) and norm(a) != norm(e['name']) and a not in e['aliases']:
                e['aliases'].append(a)
    n_raw = len(ents)
    lab = collections.Counter(norm(e['name']) for e in ents.values())
    ents = {u: e for u, e in ents.items() if lab[norm(e['name'])] == 1}
    n_dup = n_raw - len(ents)
    # alias ambiguity: a normalized alias equal to another entity's label or alias
    owner = collections.defaultdict(set)
    for u, e in ents.items():
        owner[norm(e['name'])].add(u)
        for a in e['aliases']:
            owner[norm(a)].add(u)
    n_al0 = sum(len(e['aliases']) for e in ents.values())
    for u, e in ents.items():
        keep, seen = [], set()
        for a in e['aliases']:
            na = norm(a)
            if owner[na] != {u} or na in seen:
                continue
            seen.add(na)
            keep.append(a)
        e['aliases'] = keep
    n_al1 = sum(len(e['aliases']) for e in ents.values())
    out = []
    for u in sorted(ents):
        e = ents[u]
        e['split'] = split_of(u)
        out.append(e)
    json.dump({'source': RAW, 'n_raw_entities': n_raw, 'n_dropped_duplicate_label': n_dup,
               'n_aliases_before_ambiguity_filter': n_al0, 'n_aliases': n_al1, 'entities': out},
              open(OUT, 'w'), ensure_ascii=False, indent=0)
    print(f'{n_raw} entities, {n_dup} dropped (shared label), aliases {n_al0} -> {n_al1} after ambiguity filter')
    stats()


def stats():
    d = json.load(open(OUT))
    c = collections.Counter()
    for e in d['entities']:
        c[e['split'], 'ent'] += 1
        c[e['split'], 'ent_alias'] += bool(e['aliases'])
        c[e['split'], 'alias'] += len(e['aliases'])
    for s in ('train', 'dev', 'test'):
        print(f'{s:5s} entities {c[s, "ent"]:5d}  with alias {c[s, "ent_alias"]:5d}  aliases {c[s, "alias"]:5d}')


if __name__ == '__main__':
    {'build': build, 'stats': stats}[sys.argv[1]]()
