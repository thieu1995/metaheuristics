# Improved pathfinder algorithm
```code 
- Các mô hình cần chạy :

Evolution :

    + GA 
ga_paras = {
    "epoch": 500,
    "pop_size": 100,
    "pc": 0.95,
    "pm": 0.025
}

    + DE
de_paras = {
    "epoch": 500,
    "pop_size": 100,
    "Wf": 0.8,
    "Cr": 0.9
}

Human-based:

    + QSO
qso_paras = {
    "epoch": 500,
    "pop_size": 100,
}

Physics-based:

    + TWO
two_paras = {
    "epoch": 500,
    "pop_size": 100,
}

    + NRO
nro_paras = {
    "epoch": 500,
    "pop_size": 100
}

Swarm-based:

    + PSO
pso_paras = {
    "epoch": 500,
    "pop_size": 100,
    "w_minmax": [0.4, 0.9],     # [0-1] -> [0.4-0.9]      Weight of bird
    "c_minmax": [1.2, 1.2]      # [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Effecting of  local va global
}

    + WOA
woa_paras = {
    "epoch": 500,
    "pop_size": 100
}

    + CRO
cro_paras = {
    "epoch": 500,
    "pop_size": 100,
    "G": [0.02, 0.2],
    "GCR": 0.1,
    "po": 0.4,
    "Fb": 0.9,
    "Fa": 0.1,
    "Fd": 0.1,
    "Pd": 0.1,
    "k": 3
}

    + HHO
hho_paras = {
    "epoch": 500,
    "pop_size": 100
}

    + ABC
abc_paras = {
    "epoch": 500,
    "pop_size": 100,
    "couple_bees": [16, 4],               # number of bees which provided for good location and other location
    "patch_variables": [5.0, 0.985],        # patch_variables = patch_variables * patch_factor (0.985)
    "sites": [3, 1],                        # 3 bees (employed bees, onlookers and scouts), 1 good partition
}

    + PFA
pfa_paras = {
    "epoch": 500,
    "pop_size": 100
}

    + IPFA
pfa_paras = {
    "epoch": 500,
    "pop_size": 100
}

```

