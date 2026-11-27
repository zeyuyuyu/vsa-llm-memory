# B73 (equal bits) compute note 1 (about 20:26 machine clock; no fresh entity has been received; nothing re-chosen)

- **Header time.** The pre-registration's header says "about 20:25"; it was committed at 20:21:58 (b8246b4) and stamped,
  the proof committed at 20:22:00 (02c0d07).
- **First fetch refused.** b73_data.py fetch started at about 20:22 and sent its 18 requests (9 classes x the registered
  half-size retry) back to back through python's urllib; the endpoint refused every one (HTTP 403 "Too Many Reqs", then
  429). No response body with data was received (log kept: b73_out/fresh/fetch_log_attempt1_throttled.json, not
  committed). A trivial ASK query sent with curl and the probe's User-Agent ('research-script/0.1 (generic)') at about
  20:24 was answered (HTTP 200).
- **Change (transport only).** The fetch now sends the SAME queries (same classes, sample sizes, filters, retry rule)
  through curl with that generic User-Agent, one query at a time, 15 s between queries (30 s after a failure). The
  selection rule, quotas, filters and everything downstream are unchanged. Committed with this note before the fetch is
  re-run.
