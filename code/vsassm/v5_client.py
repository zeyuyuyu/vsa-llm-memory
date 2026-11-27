"""V5 orchestrator client: one chat-completions call to gpt-6.1-sol through the user's api_zheng gateway.

Usage: python3 v5_client.py PROMPT_FILE OUT_FILE TAG [--effort high] [--max 64000]
  PROMPT_FILE: a JSON list of chat messages [{"role": ..., "content": ...}, ...]
  OUT_FILE:    the assistant's reply text, written verbatim
Every request and response (minus the token) is appended to v5_out/gpt_log.jsonl with the call's token usage.
Never api.openai.com; the gateway's credentials come from apps/real/api_reader.zheng_creds(), which parses the client
file's base64 constants without executing it. One attempt per call: a connection error is logged and the script exits
non-zero (no retry loop).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'apps', 'real'))
from api_reader import zheng_creds  # noqa: E402

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v5_out', 'gpt_log.jsonl')


def main():
    pf, outf, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    effort = 'high'
    mx = 64000
    if '--effort' in sys.argv:
        effort = sys.argv[sys.argv.index('--effort') + 1]
    if '--max' in sys.argv:
        mx = int(sys.argv[sys.argv.index('--max') + 1])
    msgs = json.load(open(pf))
    url, tok = zheng_creds()
    assert 'api.openai.com' not in url
    body = {'model': 'gpt-6.1-sol', 'messages': msgs, 'max_completion_tokens': mx, 'reasoning_effort': effort,
            'stream': True, 'stream_options': {'include_usage': True}}
    req = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Authorization': f'Bearer {tok}', 'Content-Type': 'application/json',
                                          'User-Agent': 'research-client'})
    t_start = time.time()
    while True:  # only HTTP 429 (request not processed) is waited on, per Retry-After, for at most 20 minutes
        t0 = time.time()
        rec = {'tag': tag, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'request': body}
        try:
            r = _stream(urllib.request.urlopen(req, timeout=1800))
            break
        except urllib.error.HTTPError as e:
            ra = e.headers.get('Retry-After')
            rec.update(error=f'HTTP {e.code}', error_body=e.read().decode(errors='replace')[:4000],
                       retry_after=ra, retry_after_ms=e.headers.get('retry-after-ms'),
                       seconds=round(time.time() - t0, 1))
            _log(rec)
            if e.code == 429 and time.time() - t_start < 1200:
                time.sleep(max(5.0, float(ra or 10)) + 1.0)
                continue
            print('GATEWAY HTTP ERROR', e.code, file=sys.stderr)
            sys.exit(2)
        except Exception as e:  # connection errors: log, stop, no retry
            rec.update(error=f'{type(e).__name__}: {e}', seconds=round(time.time() - t0, 1))
            _log(rec)
            print('GATEWAY CONNECTION ERROR', type(e).__name__, e, file=sys.stderr)
            sys.exit(3)
    rec.update(response=r, usage=r.get('usage'), seconds=round(time.time() - t0, 1))
    _log(rec)
    txt = r['choices'][0]['message']['content']
    open(outf, 'w').write(txt)
    print(json.dumps({'tag': tag, 'usage': r.get('usage'), 'model': r.get('model'), 'seconds': rec['seconds'],
                      'finish': r['choices'][0].get('finish_reason')}))


def _stream(resp):
    """Assemble a server-sent-event stream into one chat-completion dict (content, finish_reason, usage, model)."""
    parts, usage, model, finish, rid = [], None, None, None, None
    for raw in resp:
        line = raw.decode('utf-8', errors='replace').strip()
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            break
        ev = json.loads(data)
        model = ev.get('model') or model
        rid = ev.get('id') or rid
        if ev.get('usage'):
            usage = ev['usage']
        for ch in ev.get('choices') or []:
            delta = ch.get('delta') or {}
            if delta.get('content'):
                parts.append(delta['content'])
            if ch.get('finish_reason'):
                finish = ch['finish_reason']
    return {'id': rid, 'model': model, 'usage': usage,
            'choices': [{'message': {'role': 'assistant', 'content': ''.join(parts)}, 'finish_reason': finish}]}


def _log(rec):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
