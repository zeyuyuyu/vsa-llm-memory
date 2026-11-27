"""One conversation turn: python3 v5_turn.py CONV.json USER_MSG_FILE TAG [--effort low] [--max 20000]
CONV.json holds the message list; the user message is appended, the gateway is called (v5_client.py), and the
assistant reply is appended to CONV.json. Exit code: the client's."""
import json, subprocess, sys, os
conv, umf, tag = sys.argv[1], sys.argv[2], sys.argv[3]
rest = sys.argv[4:]
msgs = json.load(open(conv))
msgs.append({'role': 'user', 'content': open(umf).read()})
tmp = conv + '.pending.json'
json.dump(msgs, open(tmp, 'w'), indent=1)
out = conv + f'.{tag}.reply.md'
rc = subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v5_client.py'),
                      tmp, out, tag] + rest)
if rc == 0:
    msgs.append({'role': 'assistant', 'content': open(out).read()})
    json.dump(msgs, open(conv, 'w'), indent=1)
    os.remove(tmp)
sys.exit(rc)
