b35_run.py --arm vsa --codes rand --proj mlp --lr 2e-4 --mem_lr 2e-3 --B 8 --seed 0 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_vsa_B8_s0
b35_run.py --arm lmt --lmt_layers 2 --lr 2e-4 --mem_lr 2e-3 --B 8 --seed 0 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_lmt_B8_s0
b35_run.py --arm vsa --codes rand --proj mlp --lr 2e-4 --mem_lr 2e-3 --B 4 --seed 0 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_vsa_B4_s0
b35_run.py --arm lmt --lmt_layers 2 --lr 2e-4 --mem_lr 2e-3 --B 4 --seed 0 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_lmt_B4_s0
b35_run.py --arm vsa --codes rand --proj mlp --lr 2e-4 --mem_lr 2e-3 --B 2 --seed 0 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_vsa_B2_s0
b35_run.py --arm lmt --lmt_layers 2 --lr 2e-4 --mem_lr 2e-3 --B 2 --seed 0 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_lmt_B2_s0
b35_run.py --arm vsa --codes rand --proj mlp --lr 2e-4 --mem_lr 2e-3 --B 1 --seed 0 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_vsa_B1_s0
b35_run.py --arm lmt --lmt_layers 2 --lr 2e-4 --mem_lr 2e-3 --B 1 --seed 0 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_lmt_B1_s0
b35_run.py --arm ttrunc --lr 2e-4 --B 8 --seed 0 --steps 2000 --eval_splits dev,test --out out/test_ttrunc_B8_s0
b35_run.py --arm ttrunc --lr 2e-4 --B 4 --seed 0 --steps 2000 --eval_splits dev,test --out out/test_ttrunc_B4_s0
b35_run.py --arm ttrunc --lr 2e-4 --B 2 --seed 0 --steps 2000 --eval_splits dev,test --out out/test_ttrunc_B2_s0
b35_run.py --arm ttrunc --lr 2e-4 --B 1 --seed 0 --steps 2000 --eval_splits dev,test --out out/test_ttrunc_B1_s0
b35_run.py --arm tfull --lr 2e-4 --seed 0 --steps 300 --eval_splits dev,test --out out/test_tfull_s0
b35_run.py --arm tfull --lora_r 0 --steps 0 --max_new 16 --eval_splits dev,test --out out/test_tfull_zeroshot
b35_run.py --arm vsa --codes rand --proj mlp --lr 2e-4 --mem_lr 2e-3 --B 8 --seed 1 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_vsa_B8_s1
b35_run.py --arm lmt --lmt_layers 2 --lr 2e-4 --mem_lr 2e-3 --B 8 --seed 1 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_lmt_B8_s1
b35_run.py --arm vsa --codes rand --proj mlp --lr 2e-4 --mem_lr 2e-3 --B 4 --seed 1 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_vsa_B4_s1
b35_run.py --arm lmt --lmt_layers 2 --lr 2e-4 --mem_lr 2e-3 --B 4 --seed 1 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_lmt_B4_s1
b35_run.py --arm vsa --codes rand --proj mlp --lr 2e-4 --mem_lr 2e-3 --B 2 --seed 1 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_vsa_B2_s1
b35_run.py --arm lmt --lmt_layers 2 --lr 2e-4 --mem_lr 2e-3 --B 2 --seed 1 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_lmt_B2_s1
b35_run.py --arm vsa --codes rand --proj mlp --lr 2e-4 --mem_lr 2e-3 --B 1 --seed 1 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_vsa_B1_s1
b35_run.py --arm lmt --lmt_layers 2 --lr 2e-4 --mem_lr 2e-3 --B 1 --seed 1 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_lmt_B1_s1
b35_run.py --arm vsaq --codes rand --proj mlp --lr 2e-4 --mem_lr 2e-3 --B 8 --seed 0 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_vsaq_B8_s0
b35_run.py --arm vsaq --codes rand --proj mlp --lr 2e-4 --mem_lr 2e-3 --B 4 --seed 0 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_vsaq_B4_s0
b35_run.py --arm vsaq --codes rand --proj mlp --lr 2e-4 --mem_lr 2e-3 --B 2 --seed 0 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_vsaq_B2_s0
b35_run.py --arm vsaq --codes rand --proj mlp --lr 2e-4 --mem_lr 2e-3 --B 1 --seed 0 --steps 8000 --eval_every 2000 --eval_sub 100 --eval_splits dev,test --out out/test_vsaq_B1_s0
