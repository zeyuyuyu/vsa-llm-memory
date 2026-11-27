"""B35 (prereg/B35_soft_token_vsa_prereg.md): synthetic entity-attribute facts for the soft-token pilot.

An instance holds N facts about E = N / 4 invented people. Each person has three attributes (drawn without
replacement from six: city, color, pet, sport, instrument, food; 32 single-word values each, drawn uniformly) and a
manager (another person of the same instance). Facts are written as sentences ("The city of Velor is Lisbon.",
"The manager of Velor is Quenna.") in a random order. Every instance carries one single-hop question (an attribute
of a person) and one two-hop question (an attribute of the person's manager; the manager always has it).

Names come from one fixed pool of 600 invented names shared by all splits (an instance is a fresh random draw of
people, attributes, values and managers, so no fact can be memorised across instances). Everything is
deterministic given (split, N, idx).
"""
import hashlib
import random

ATTRS = {
    'city': 'Paris Tokyo Lisbon Cairo Madrid Berlin Rome Vienna Oslo Dublin Prague Athens Warsaw Seoul Lima Quito '
            'Nairobi Delhi Boston Denver Chicago Toronto Sydney Moscow Istanbul Bangkok Manila Havana Helsinki '
            'Zurich Milan Munich'.split(),
    'color': 'red blue green yellow purple orange pink brown black white gray teal crimson violet indigo maroon '
             'beige gold silver navy olive coral amber ivory lime cyan magenta turquoise lavender scarlet bronze '
             'khaki'.split(),
    'pet': 'cat dog rabbit hamster parrot turtle goldfish ferret lizard snake mouse rat pony goat pig duck chicken '
           'frog owl hedgehog canary gecko iguana tortoise donkey sheep llama alpaca crab squirrel raccoon '
           'chinchilla'.split(),
    'sport': 'tennis soccer golf rugby hockey cricket baseball basketball volleyball boxing fencing rowing cycling '
             'skiing surfing swimming archery badminton karate judo wrestling sailing climbing running diving '
             'skating bowling squash polo lacrosse handball snooker'.split(),
    'instrument': 'piano violin guitar cello flute trumpet drums harp clarinet saxophone oboe banjo ukulele '
                  'accordion trombone tuba viola bassoon harmonica mandolin xylophone sitar organ bagpipes lute '
                  'piccolo tambourine marimba zither kazoo cornet recorder'.split(),
    'food': 'pizza sushi pasta tacos curry soup salad steak noodles dumplings burgers pancakes waffles risotto '
            'paella lasagna ramen falafel kebab burritos omelette porridge chowder goulash couscous gnocchi hummus '
            'nachos pretzels croissants bagels muffins'.split(),
}
ATTR_NAMES = list(ATTRS)
REL = 'manager'
RELATIONS = ATTR_NAMES + [REL]
NS = (8, 32, 128)
N_NAMES = 600


def seed_of(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


def _make_names():
    rng = random.Random(seed_of('b35', 'names'))
    on = ['b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'z', 'br', 'dr', 'kr', 'tr', 'th',
          'sh', 'gr', 'pl', 'qu', 'h', 'j']
    vo = ['a', 'e', 'i', 'o', 'u', 'ai', 'ei', 'ou', 'ia', 'ae']
    co = ['', '', '', 'n', 'r', 'l', 's', 'th', 'x', 'm']
    try:
        words = {w.strip().lower() for w in open('/usr/share/dict/words')}
    except OSError:
        words = set()
    vals = {v.lower() for vs in ATTRS.values() for v in vs}
    out, seen = [], set()
    while len(out) < N_NAMES:
        k = rng.choice([2, 2, 3])
        s = ''.join(rng.choice(on) + rng.choice(vo) for _ in range(k)) + rng.choice(co)
        if not (4 <= len(s) <= 9) or s in words or s in vals or s in seen:
            continue
        seen.add(s)
        out.append(s.capitalize())
    return out


NAMES = _make_names()
assert len(set(NAMES)) == N_NAMES
_ALLV = [v for a in ATTR_NAMES for v in ATTRS[a]]
assert len(set(_ALLV)) == len(_ALLV) == 192 and not set(NAMES) & set(_ALLV)


def fact_text(e, r, v):
    return f'The {r} of {e} is {v}.'


def instance(split, N, idx):
    """Return dict: facts [(e, r, v)] in text order, text, q1 (single hop), q2 (two hop)."""
    assert N % 4 == 0 and N >= 8
    rng = random.Random(seed_of('b35', split, N, idx))
    E = N // 4
    people = rng.sample(NAMES, E)
    facts, attrs, mgr = [], {}, {}
    for i, p in enumerate(people):
        attrs[p] = {}
        for a in rng.sample(ATTR_NAMES, 3):
            v = rng.choice(ATTRS[a])
            attrs[p][a] = v
            facts.append((p, a, v))
        j = rng.randrange(E - 1)
        m = people[j if j < i else j + 1]
        mgr[p] = m
        facts.append((p, REL, m))
    rng.shuffle(facts)
    x = rng.choice(people)
    a = rng.choice(sorted(attrs[x]))
    q1 = dict(qtype='hop1', text=f'What is the {a} of {x}?', answer=attrs[x][a], chain=[(x, a)])
    x2 = rng.choice(people)
    y = mgr[x2]
    a2 = rng.choice(sorted(attrs[y]))
    q2 = dict(qtype='hop2', text=f'What is the {a2} of the manager of {x2}?', answer=attrs[y][a2],
              chain=[(x2, REL), (y, a2)])
    return dict(split=split, N=N, idx=idx, facts=facts, text=' '.join(fact_text(*f) for f in facts), q1=q1, q2=q2)


def dict_answer(inst, q):
    """Exact external lookup (the DICT arm): follow the question's chain through a dictionary of the facts."""
    d = {(e, r): v for e, r, v in inst['facts']}
    # parse the question text itself (not the stored chain) with the two fixed templates
    t = q['text']
    if t.startswith('What is the ') and ' of the manager of ' in t:
        a, x = t[len('What is the '):-1].split(' of the manager of ')
        return d.get((d.get((x, REL)), a))
    a, x = t[len('What is the '):-1].split(' of ', 1)
    return d.get((x, a))


SPLIT_SIZES = {'dev': 300, 'test': 600}


def eval_set(split):
    return [instance(split, N, i) for N in NS for i in range(SPLIT_SIZES[split])]


if __name__ == '__main__':
    import sys
    print(NAMES[:20])
    for sp in ('dev', 'test'):
        es = eval_set(sp)
        ok = sum(dict_answer(x, x[q]) == x[q]['answer'] for x in es for q in ('q1', 'q2'))
        print(sp, len(es), 'instances; DICT correct', ok, '/', 2 * len(es))
    x = instance('dev', 8, 0)
    print(x['text']); print(x['q1']); print(x['q2'])
    sys.exit(0)
