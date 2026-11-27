#!/bin/sh
# One command: recompute the manuscript's headline numbers from the archived outputs (Python 3 with numpy).
set -e
cd "$(dirname "$0")"
python3 equal_bit.py
echo; python3 device_cells.py
echo; python3 forward_pass.py
