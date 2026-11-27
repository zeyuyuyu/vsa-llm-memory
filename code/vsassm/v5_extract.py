"""Extract the single ```python block of a GPT reply verbatim: python3 v5_extract.py REPLY.md OUT.py"""
import re
import sys

txt = open(sys.argv[1]).read()
blocks = re.findall(r"```python\n(.*?)```", txt, flags=re.S)
if len(blocks) != 1:
    print(f"found {len(blocks)} python blocks", file=sys.stderr)
    if not blocks:
        sys.exit(1)
open(sys.argv[2], 'w').write(blocks[0] if len(blocks) == 1 else max(blocks, key=len))
print(sys.argv[2], len(blocks[0].splitlines()), 'lines')
