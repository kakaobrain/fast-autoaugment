#!/usr/bin/env bash
set -euo pipefail
sleep infinity || [ $? -eq 143 ]
