"""Apply a GPT fix reply mechanically: python3 v5_patch.py OLD.py REPLY.md NEW.py
The reply's single ```python block holds complete top-level definitions (def / class / NAME = ... assignments). Each one
replaces the top-level node of the same name in OLD.py, in place; a name not present is inserted before the first
top-level `if __name__ ...` block (or at the end). No other edits are made."""
import ast, re, sys
old = open(sys.argv[1]).read()
rep = open(sys.argv[2]).read()
blocks = re.findall(r"```python\n(.*?)```", rep, flags=re.S)
assert len(blocks) == 1, f'{len(blocks)} python blocks'
new_src = blocks[0]
def names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []
nt = ast.parse(new_src)
nlines = new_src.splitlines(keepends=True)
ot = ast.parse(old)
olines = old.splitlines(keepends=True)
def span(node, lines):
    start = node.lineno - 1
    if getattr(node, 'decorator_list', None):
        start = min(d.lineno for d in node.decorator_list) - 1
    return start, node.end_lineno
repl, extra = {}, []
for node in nt.body:
    nm = names(node)
    s, e = span(node, nlines)
    text = ''.join(nlines[s:e])
    if nm:
        repl[nm[0]] = text
    else:
        extra.append(text)
done = set()
out, i = [], 0
nodes = sorted(ot.body, key=lambda n: n.lineno)
for node in nodes:
    s, e = span(node, olines)
    nm = names(node)
    if nm and nm[0] in repl and nm[0] not in done:
        out.append(''.join(olines[i:s]))
        out.append(repl[nm[0]].rstrip('\n') + '\n')
        done.add(nm[0]); i = e
out.append(''.join(olines[i:]))
res = ''.join(out)
missing = [k for k in repl if k not in done]
if missing or extra:
    ins = ''.join(repl[k].rstrip('\n') + '\n\n\n' for k in missing) + ''.join(x.rstrip('\n') + '\n\n' for x in extra)
    m = re.search(r"^if __name__ == ['\"]__main__['\"]:", res, flags=re.M)
    res = res[:m.start()] + ins + res[m.start():] if m else res + '\n\n' + ins
ast.parse(res)
open(sys.argv[3], 'w').write(res)
print('replaced', sorted(done), 'inserted', missing, 'extra statements', len(extra))
