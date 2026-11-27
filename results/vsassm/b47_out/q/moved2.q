b47_ft.py --task gen --variant std --n 8000 --rank 64 --lr 3e-05 --seed 123 --final --mbs 4 --out out/ft/gen_std_n8000_r64_lr3e-05_s123.json
b47_ft.py --task fullgen --variant std --n 8000 --rank 8 --lr 3e-06 --seed 123 --final --mbs 8 --out out/ft/fullgen_std_n8000_r8_lr3e-06_s123.json
b47_ft.py --task scratch --variant k32 --n 800 --rank 8 --lr 0.0003 --seed 456 --final --mbs 4 --out out/ft/scratch_k32_n800_r8_lr0.0003_s456.json
b47_ft.py --task gen --variant std --n 8000 --rank 64 --lr 3e-05 --seed 456 --final --mbs 4 --out out/ft/gen_std_n8000_r64_lr3e-05_s456.json
