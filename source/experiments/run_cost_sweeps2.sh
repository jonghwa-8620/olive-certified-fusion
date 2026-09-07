#!/bin/bash
# 2단계: 나머지 보조 실험을 비용 단위로 재실행한다.
export PYTHONIOENCODING=utf-8
export LACF_UNITS=cost
cd "$(dirname "$0")"
set -x
python3 lacf_ceiling.py    --views runs/*__b100 --out out_cost                      > logs/cu_ceiling.log 2>&1
python3 lacf_gate.py       --views runs/*__b100 --base product_rule --out out_cost  > logs/cu_gate.log 2>&1
python3 lacf_base.py       --views runs/*__b100 --out out_cost                      > logs/cu_base.log 2>&1
python3 lacf_automation.py --views runs/*__b100 --base product_rule --out out_cost  > logs/cu_auto.log 2>&1
python3 lacf_estimation.py --views runs/*__b100 --base product_rule --out out_cost  > logs/cu_est.log 2>&1
python3 lacf_costmat.py    --views runs/*__b100 --base product_rule --out out_cost  > logs/cu_costmat.log 2>&1
echo DONE2
