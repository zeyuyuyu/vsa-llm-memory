#!/bin/bash
# B44: ssh to the A800 box through one persistent master connection (fewer new connections; the box resets some).
exec ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm/b44-%r@%h:%p -o ControlPersist=30m -o ConnectTimeout=40 cavin "$@"
