#!/bin/bash
# Theorem 4 정의(비용 단위)로 보조 스윕을 다시 돌린다.
export PYTHONIOENCODING=utf-8
export LACF_UNITS=cost
cd "$(dirname "$0")"
set -x
python3 sliver_referral.py   --views runs/*__b100 --out out_cost                       > logs/cu_referral.log 2>&1
python3 sliver_arithmetic.py --views runs/*__b100 --base product_rule --out out_cost \
        --ks 0.5 1 2 3 4 6 8 12 20                                                     > logs/cu_arith.log 2>&1
python3 lacf_sliver_k.py     --views runs/*__b100 --base product_rule --out out_cost \
        --ks 0.5 1 2 3 4 6                                                             > logs/cu_k.log 2>&1
echo DONE
