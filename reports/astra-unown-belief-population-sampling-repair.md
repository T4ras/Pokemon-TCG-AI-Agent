> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Population-based sampling and neural deck-head regression

State: **COMPLETE**. Updated [removed].

Allfour runs completed exactly14004updates each. Fixed-final comparisons, initialization checks, secondary selection and exposure diagnostics saved; no test access.

This registered comparison changes only how population prevalence enters one-BO3 updates. It uses all 2,334 training histories as the eligible pool and the same 592 validation histories. The 592 previously opened test histories and their predictions are excluded; they are no longer an untouched test. No native matches, GPU, PPO, paid model APIs, uploads, submissions, tuning or automatic follow-up. Historical evidence is preserved.

## Frozen comparison

Both seeds (11 and 29) use the existing 57,691-parameter zero-residual MLP. Within each seed, control and treatment load exactly the same initial model, fresh AdamW state and Torch RNG state. Initialization seeds and sampling seeds are recorded separately. AdamW remains lr 0.001, weight decay 0.0001, betas (0.9,0.999), epsilon 1e-8; global gradient clipping stays 1. One complete BO3 supplies each update. Features, masks and the original loss are unchanged: deck CE plus the mean of hand/draw-pile/prize count CE, with query mean, decision mean within game and equal game weight within a match.

Control makes six shuffled passes through all histories, multiplying the match loss by N p(deck)/(2 n(deck,chooser)). Treatment independently samples a deck by frozen n/100 prevalence, a chooser with probability 1/2, then a history uniformly with replacement within that stratum; its match weight is one. All histories remain eligible. No second prevalence correction is applied.

For any fixed per-history loss L, both expected scalar objectives equal Σ_d p(d)/2 Σ_c mean[L | d,c]. The coefficient and outcome-independent sampler checks are in sampler-check.json (historical/private artifact omitted). This identity does not imply equivalent updates through AdamW and clipping.

Each of four runs receives exactly 14,004 updates, with checkpoints at 0, 2,334, 4,668, 7,002, 9,336, 11,670 and 14,004. No early stopping. The primary endpoint is **full-field n/100 validation deck NLL at 14,004**, treatment minus matched control; negative is better. The n/82 passing-opponent population is separately labelled.

The prelaunch conservative forecast was 3.569 CPU-core hours against the four-hour cap, including historical worst epoch/evaluation timings, stationary replays and overhead. Six UTC elapsed hours and four compute threads remain hard caps; the runner retains failures and stops rather than reducing the protocol.

## Execution and exposure

| Run | State | Updates | Unique / eligible histories | CPU minutes |
|---|---|---:|---:|---:|
| control-s11 | COMPLETE | 14004 | 2334 / 2334 | 25.20 |
| control-s29 | COMPLETE | 14004 | 2334 / 2334 | 24.73 |
| population-s11 | COMPLETE | 14004 | 2167 / 2334 | 25.15 |
| population-s29 | COMPLETE | 14004 | 2165 / 2334 | 24.84 |

Realized exposure counts and every deck/chooser frequency, including histories with zero exposure, are retained in exposures (historical/private artifact omitted). Control visits every history six times when complete; treatment coverage is a measured outcome of its fixed sampling schedule, not a requirement to visit each record.

| Run | Chooser 0 / 1 updates | Maximum absolute deck-frequency deviation from p(deck) |
|---|---:|---:|
| control-s11 | 6978 / 7026 | 0.184293 |
| control-s29 | 6978 / 7026 | 0.184293 |
| population-s11 | 6995 / 7009 | 0.004010 |
| population-s29 | 6943 / 7061 | 0.005533 |

Control exposure frequencies follow the nearly uniform collection; its loss weights represent prevalence. Treatment exposure frequencies directly represent prevalence. These frequencies are therefore expected to differ.

## Primary fixed-final comparison

Paired whole-BO3 bootstrap draws are shared across methods within deck×chooser strata (10,000 replicates). Intervals condition on these fitted models, do not cover all training-seed uncertainty, and are not multiplicity-adjusted simultaneous claims. Seed means average scores within BO3; they are not ensembles or extra independent matches.

| Population | Seed | Treatment − control deck NLL, 95% interval |
|---|---|---:|
| Full field n/100 — primary | s11 | -0.392827 [-0.466323, -0.311563] |
| Full field n/100 — primary | s29 | -0.253121 [-0.315445, -0.186447] |
| Full field n/100 — primary | seedmean | -0.322974 [-0.390359, -0.249041] |
| Passing opponents n/82 | s11 | -0.479988 [-0.569507, -0.380689] |
| Passing opponents n/82 | s29 | -0.320393 [-0.396090, -0.239031] |
| Passing opponents n/82 | seedmean | -0.400191 [-0.482087, -0.309879] |

Treatment improves fixed-final deckNLL over matched control on these fitted seeds.

### Treatment versus initialization and A

| Reference | Seed | Final treatment − reference deck NLL (n/100), 95% interval |
|---|---|---:|
| initial | s11 | -0.021563 [-0.036084, -0.007743] |
| initial | s29 | +0.002450 [-0.017340, +0.022093] |
| initial | seedmean | -0.009556 [-0.026571, +0.006974] |
| A | s11 | -0.021563 [-0.036084, -0.007743] |
| A | s29 | +0.002450 [-0.017340, +0.022093] |
| A | seedmean | -0.009556 [-0.026571, +0.006974] |

A treatment that only regresses less than control is optimization evidence, not a successful replacement for A. Gains against control, initialization and A must be assessed separately. Conflicting seeds and intervals containing zero remain unresolved.

## Full field n/100: initialization and fixed final

| Model/checkpoint | Deck NLL | Deck accuracy | Aggregate count NLL | Combined objective |
|---|---:|---:|---:|---:|
| A | 0.539081 [0.509868, 0.569065] | 0.820823 [0.806504, 0.834954] | 0.098519 [0.097648, 0.099417] | 0.637600 [0.608101, 0.667893] |
| control-s11-u00000 | 0.539081 [0.509868, 0.569065] | 0.820823 [0.806504, 0.834954] | 0.098519 [0.097648, 0.099417] | 0.637600 [0.608101, 0.667893] |
| control-s29-u00000 | 0.539081 [0.509868, 0.569065] | 0.820823 [0.806504, 0.834954] | 0.098519 [0.097648, 0.099417] | 0.637600 [0.608101, 0.667893] |
| control-seedmean-u00000 | 0.539081 [0.509868, 0.569065] | 0.820823 [0.806504, 0.834954] | 0.098519 [0.097648, 0.099417] | 0.637600 [0.608101, 0.667893] |
| population-s11-u00000 | 0.539081 [0.509868, 0.569065] | 0.820823 [0.806504, 0.834954] | 0.098519 [0.097648, 0.099417] | 0.637600 [0.608101, 0.667893] |
| population-s29-u00000 | 0.539081 [0.509868, 0.569065] | 0.820823 [0.806504, 0.834954] | 0.098519 [0.097648, 0.099417] | 0.637600 [0.608101, 0.667893] |
| population-seedmean-u00000 | 0.539081 [0.509868, 0.569065] | 0.820823 [0.806504, 0.834954] | 0.098519 [0.097648, 0.099417] | 0.637600 [0.608101, 0.667893] |
| control-s11-u14004 | 0.910345 [0.822072, 0.991973] | 0.643379 [0.617660, 0.672954] | 0.097304 [0.096480, 0.098171] | 1.007649 [0.918863, 1.089404] |
| control-s29-u14004 | 0.794652 [0.718904, 0.864982] | 0.671753 [0.644532, 0.702557] | 0.096990 [0.096144, 0.097869] | 0.891641 [0.815599, 0.962314] |
| control-seedmean-u14004 | 0.852498 [0.771107, 0.928204] | 0.657566 [0.631883, 0.687339] | 0.097147 [0.096314, 0.098022] | 0.949645 [0.867620, 1.025409] |
| population-s11-u14004 | 0.517518 [0.485901, 0.550121] | 0.834014 [0.819902, 0.848101] | 0.096469 [0.095732, 0.097228] | 0.613987 [0.582034, 0.646982] |
| population-s29-u14004 | 0.541531 [0.504902, 0.578796] | 0.833211 [0.818975, 0.847226] | 0.096489 [0.095715, 0.097276] | 0.638020 [0.601054, 0.675600] |
| population-seedmean-u14004 | 0.529525 [0.495564, 0.564126] | 0.833613 [0.819566, 0.847647] | 0.096479 [0.095725, 0.097248] | 0.626003 [0.591727, 0.660922] |

### Count NLL

| Model/checkpoint | Unknown hand | Draw pile | Prizes |
|---|---:|---:|---:|
| A | 0.090168 [0.089139, 0.091189] | 0.129482 [0.127932, 0.131148] | 0.075907 [0.075083, 0.076753] |
| control-s11-u00000 | 0.090168 [0.089139, 0.091189] | 0.129482 [0.127932, 0.131148] | 0.075907 [0.075083, 0.076753] |
| control-s29-u00000 | 0.090168 [0.089139, 0.091189] | 0.129482 [0.127932, 0.131148] | 0.075907 [0.075083, 0.076753] |
| control-seedmean-u00000 | 0.090168 [0.089139, 0.091189] | 0.129482 [0.127932, 0.131148] | 0.075907 [0.075083, 0.076753] |
| population-s11-u00000 | 0.090168 [0.089139, 0.091189] | 0.129482 [0.127932, 0.131148] | 0.075907 [0.075083, 0.076753] |
| population-s29-u00000 | 0.090168 [0.089139, 0.091189] | 0.129482 [0.127932, 0.131148] | 0.075907 [0.075083, 0.076753] |
| population-seedmean-u00000 | 0.090168 [0.089139, 0.091189] | 0.129482 [0.127932, 0.131148] | 0.075907 [0.075083, 0.076753] |
| control-s11-u14004 | 0.088188 [0.087246, 0.089130] | 0.128936 [0.127276, 0.130704] | 0.074789 [0.074013, 0.075576] |
| control-s29-u14004 | 0.087810 [0.086870, 0.088741] | 0.128631 [0.126926, 0.130456] | 0.074528 [0.073723, 0.075356] |
| control-seedmean-u14004 | 0.087999 [0.087060, 0.088930] | 0.128784 [0.127099, 0.130573] | 0.074659 [0.073870, 0.075470] |
| population-s11-u14004 | 0.087217 [0.086315, 0.088110] | 0.127720 [0.126227, 0.129246] | 0.074469 [0.073672, 0.075290] |
| population-s29-u14004 | 0.087087 [0.086182, 0.087961] | 0.127898 [0.126364, 0.129486] | 0.074483 [0.073672, 0.075298] |
| population-seedmean-u14004 | 0.087152 [0.086255, 0.088032] | 0.127809 [0.126302, 0.129366] | 0.074476 [0.073683, 0.075295] |

### All-card presence Brier

| Model/checkpoint | Unknown hand | Draw pile | Prizes |
|---|---:|---:|---:|
| A | 0.023989 [0.023739, 0.024231] | 0.018188 [0.017788, 0.018594] | 0.021778 [0.021519, 0.022036] |
| control-s11-u00000 | 0.023989 [0.023739, 0.024231] | 0.018188 [0.017788, 0.018594] | 0.021778 [0.021519, 0.022036] |
| control-s29-u00000 | 0.023989 [0.023739, 0.024231] | 0.018188 [0.017788, 0.018594] | 0.021778 [0.021519, 0.022036] |
| control-seedmean-u00000 | 0.023989 [0.023739, 0.024231] | 0.018188 [0.017788, 0.018594] | 0.021778 [0.021519, 0.022036] |
| population-s11-u00000 | 0.023989 [0.023739, 0.024231] | 0.018188 [0.017788, 0.018594] | 0.021778 [0.021519, 0.022036] |
| population-s29-u00000 | 0.023989 [0.023739, 0.024231] | 0.018188 [0.017788, 0.018594] | 0.021778 [0.021519, 0.022036] |
| population-seedmean-u00000 | 0.023989 [0.023739, 0.024231] | 0.018188 [0.017788, 0.018594] | 0.021778 [0.021519, 0.022036] |
| control-s11-u14004 | 0.023342 [0.023079, 0.023593] | 0.018064 [0.017690, 0.018450] | 0.021442 [0.021199, 0.021682] |
| control-s29-u14004 | 0.023228 [0.022977, 0.023474] | 0.018016 [0.017637, 0.018409] | 0.021370 [0.021130, 0.021609] |
| control-seedmean-u14004 | 0.023285 [0.023027, 0.023532] | 0.018040 [0.017664, 0.018430] | 0.021406 [0.021166, 0.021645] |
| population-s11-u14004 | 0.023061 [0.022806, 0.023309] | 0.017797 [0.017404, 0.018196] | 0.021393 [0.021151, 0.021637] |
| population-s29-u14004 | 0.023010 [0.022754, 0.023257] | 0.017804 [0.017411, 0.018207] | 0.021403 [0.021164, 0.021646] |
| population-seedmean-u14004 | 0.023035 [0.022781, 0.023281] | 0.017801 [0.017405, 0.018203] | 0.021398 [0.021157, 0.021640] |

### Fixed A-defined uncertain presence Brier

| Model/checkpoint | Unknown hand | Draw pile | Prizes |
|---|---:|---:|---:|
| A | 0.182978 [0.181129, 0.184814] | 0.181397 [0.177422, 0.185495] | 0.165431 [0.162894, 0.167940] |
| control-s11-u00000 | 0.182978 [0.181129, 0.184814] | 0.181397 [0.177422, 0.185495] | 0.165431 [0.162894, 0.167940] |
| control-s29-u00000 | 0.182978 [0.181129, 0.184814] | 0.181397 [0.177422, 0.185495] | 0.165431 [0.162894, 0.167940] |
| control-seedmean-u00000 | 0.182978 [0.181129, 0.184814] | 0.181397 [0.177422, 0.185495] | 0.165431 [0.162894, 0.167940] |
| population-s11-u00000 | 0.182978 [0.181129, 0.184814] | 0.181397 [0.177422, 0.185495] | 0.165431 [0.162894, 0.167940] |
| population-s29-u00000 | 0.182978 [0.181129, 0.184814] | 0.181397 [0.177422, 0.185495] | 0.165431 [0.162894, 0.167940] |
| population-seedmean-u00000 | 0.182978 [0.181129, 0.184814] | 0.181397 [0.177422, 0.185495] | 0.165431 [0.162894, 0.167940] |
| control-s11-u14004 | 0.177385 [0.175432, 0.179301] | 0.179212 [0.175505, 0.182994] | 0.162934 [0.160481, 0.165361] |
| control-s29-u14004 | 0.176447 [0.174532, 0.178344] | 0.178683 [0.174953, 0.182467] | 0.162369 [0.160017, 0.164722] |
| control-seedmean-u14004 | 0.176916 [0.174986, 0.178810] | 0.178948 [0.175232, 0.182719] | 0.162651 [0.160249, 0.165046] |
| population-s11-u14004 | 0.175378 [0.173454, 0.177332] | 0.177470 [0.173396, 0.181555] | 0.162751 [0.160353, 0.165153] |
| population-s29-u14004 | 0.175045 [0.173085, 0.177012] | 0.177374 [0.173385, 0.181413] | 0.162760 [0.160385, 0.165143] |
| population-seedmean-u14004 | 0.175211 [0.173275, 0.177140] | 0.177422 [0.173400, 0.181503] | 0.162756 [0.160373, 0.165129] |

### Final legality and count consistency

| Final model | Support mass violation | Hand / draw-pile / prize total MAE | Card-total violation MAE / fraction |
|---|---:|---|---|
| control-s11-u14004 | 0.000000000 | 0.332377 / 0.490119 / 0.204620 | 0.001814 / 0.036520 |
| control-s29-u14004 | 0.000000000 | 0.241315 / 0.409528 / 0.161497 | 0.001782 / 0.052364 |
| control-seedmean-u14004 | 0.000000000 | 0.286846 / 0.449824 / 0.183059 | 0.001798 / 0.044442 |
| population-s11-u14004 | 0.000000000 | 0.231138 / 0.374026 / 0.176040 | 0.002635 / 0.071007 |
| population-s29-u14004 | 0.000000000 | 0.242364 / 0.430889 / 0.158375 | 0.002328 / 0.055594 |
| population-seedmean-u14004 | 0.000000000 | 0.236751 / 0.402458 / 0.167207 | 0.002482 / 0.063301 |

Marginal count consistency is diagnostic; no projection or behavior correction was introduced. Hand targets exclude legally known hand cards; those cards are added back when checking zone totals. Fixed uncertain subsets use A presence probability in [0.1,0.9] separately per zone and are identical across models.

## Passing opponents n/82: initialization and fixed final

| Model/checkpoint | Deck NLL | Deck accuracy | Aggregate count NLL | Combined objective |
|---|---:|---:|---:|---:|
| A | 0.604108 [0.569201, 0.640170] | 0.799059 [0.781829, 0.816260] | 0.100303 [0.099292, 0.101313] | 0.704412 [0.668964, 0.741207] |
| control-s11-u00000 | 0.604108 [0.569201, 0.640170] | 0.799059 [0.781829, 0.816260] | 0.100304 [0.099293, 0.101313] | 0.704412 [0.668964, 0.741207] |
| control-s29-u00000 | 0.604108 [0.569201, 0.640170] | 0.799059 [0.781829, 0.816260] | 0.100304 [0.099293, 0.101313] | 0.704412 [0.668964, 0.741207] |
| control-seedmean-u00000 | 0.604108 [0.569201, 0.640170] | 0.799059 [0.781829, 0.816260] | 0.100304 [0.099293, 0.101313] | 0.704412 [0.668964, 0.741207] |
| population-s11-u00000 | 0.604108 [0.569201, 0.640170] | 0.799059 [0.781829, 0.816260] | 0.100304 [0.099293, 0.101313] | 0.704412 [0.668964, 0.741207] |
| population-s29-u00000 | 0.604108 [0.569201, 0.640170] | 0.799059 [0.781829, 0.816260] | 0.100304 [0.099293, 0.101313] | 0.704412 [0.668964, 0.741207] |
| population-seedmean-u00000 | 0.604108 [0.569201, 0.640170] | 0.799059 [0.781829, 0.816260] | 0.100304 [0.099293, 0.101313] | 0.704412 [0.668964, 0.741207] |
| control-s11-u14004 | 1.057827 [0.950734, 1.156864] | 0.581838 [0.550624, 0.617821] | 0.099237 [0.098293, 0.100227] | 1.157064 [1.049223, 1.256605] |
| control-s29-u14004 | 0.919555 [0.828800, 1.004995] | 0.616467 [0.583412, 0.653497] | 0.098907 [0.097940, 0.099913] | 1.018462 [0.927344, 1.104168] |
| control-seedmean-u14004 | 0.988691 [0.890460, 1.079931] | 0.599153 [0.567702, 0.635151] | 0.099072 [0.098122, 0.100070] | 1.087763 [0.989026, 1.179122] |
| population-s11-u14004 | 0.577839 [0.540390, 0.616582] | 0.814987 [0.797896, 0.831846] | 0.098138 [0.097327, 0.098969] | 0.675976 [0.638071, 0.715107] |
| population-s29-u14004 | 0.599162 [0.556571, 0.643019] | 0.814008 [0.796883, 0.830914] | 0.098162 [0.097301, 0.099036] | 0.697324 [0.654356, 0.741623] |
| population-seedmean-u14004 | 0.588501 [0.548626, 0.629469] | 0.814497 [0.797417, 0.831436] | 0.098150 [0.097314, 0.098999] | 0.686650 [0.646319, 0.728052] |

### Count NLL

| Model/checkpoint | Unknown hand | Draw pile | Prizes |
|---|---:|---:|---:|
| A | 0.093370 [0.092205, 0.094513] | 0.131849 [0.130061, 0.133751] | 0.075691 [0.074756, 0.076653] |
| control-s11-u00000 | 0.093370 [0.092205, 0.094513] | 0.131849 [0.130061, 0.133751] | 0.075692 [0.074756, 0.076653] |
| control-s29-u00000 | 0.093370 [0.092205, 0.094513] | 0.131849 [0.130061, 0.133751] | 0.075692 [0.074756, 0.076653] |
| control-seedmean-u00000 | 0.093370 [0.092205, 0.094513] | 0.131849 [0.130061, 0.133751] | 0.075692 [0.074756, 0.076653] |
| population-s11-u00000 | 0.093370 [0.092205, 0.094513] | 0.131849 [0.130061, 0.133751] | 0.075692 [0.074756, 0.076653] |
| population-s29-u00000 | 0.093370 [0.092205, 0.094513] | 0.131849 [0.130061, 0.133751] | 0.075692 [0.074756, 0.076653] |
| population-seedmean-u00000 | 0.093370 [0.092205, 0.094513] | 0.131849 [0.130061, 0.133751] | 0.075692 [0.074756, 0.076653] |
| control-s11-u14004 | 0.091567 [0.090495, 0.092656] | 0.131667 [0.129789, 0.133746] | 0.074477 [0.073602, 0.075373] |
| control-s29-u14004 | 0.091135 [0.090089, 0.092196] | 0.131384 [0.129431, 0.133523] | 0.074201 [0.073274, 0.075136] |
| control-seedmean-u14004 | 0.091351 [0.090298, 0.092424] | 0.131526 [0.129611, 0.133625] | 0.074339 [0.073442, 0.075251] |
| population-s11-u14004 | 0.090392 [0.089366, 0.091386] | 0.129939 [0.128309, 0.131643] | 0.074082 [0.073172, 0.075011] |
| population-s29-u14004 | 0.090261 [0.089221, 0.091261] | 0.130133 [0.128451, 0.131909] | 0.074091 [0.073178, 0.075004] |
| population-seedmean-u14004 | 0.090326 [0.089294, 0.091312] | 0.130036 [0.128387, 0.131775] | 0.074086 [0.073184, 0.074997] |

### All-card presence Brier

| Model/checkpoint | Unknown hand | Draw pile | Prizes |
|---|---:|---:|---:|
| A | 0.024701 [0.024430, 0.024962] | 0.017962 [0.017538, 0.018392] | 0.021757 [0.021478, 0.022042] |
| control-s11-u00000 | 0.024701 [0.024430, 0.024962] | 0.017962 [0.017538, 0.018392] | 0.021757 [0.021478, 0.022042] |
| control-s29-u00000 | 0.024701 [0.024430, 0.024962] | 0.017962 [0.017538, 0.018392] | 0.021757 [0.021478, 0.022042] |
| control-seedmean-u00000 | 0.024701 [0.024430, 0.024962] | 0.017962 [0.017538, 0.018392] | 0.021757 [0.021478, 0.022042] |
| population-s11-u00000 | 0.024701 [0.024430, 0.024962] | 0.017962 [0.017538, 0.018392] | 0.021757 [0.021478, 0.022042] |
| population-s29-u00000 | 0.024701 [0.024430, 0.024962] | 0.017962 [0.017538, 0.018392] | 0.021757 [0.021478, 0.022042] |
| population-seedmean-u00000 | 0.024701 [0.024430, 0.024962] | 0.017962 [0.017538, 0.018392] | 0.021757 [0.021478, 0.022042] |
| control-s11-u14004 | 0.024085 [0.023795, 0.024360] | 0.017919 [0.017521, 0.018328] | 0.021405 [0.021143, 0.021669] |
| control-s29-u14004 | 0.023956 [0.023675, 0.024223] | 0.017872 [0.017471, 0.018282] | 0.021323 [0.021063, 0.021586] |
| control-seedmean-u14004 | 0.024021 [0.023736, 0.024291] | 0.017895 [0.017498, 0.018306] | 0.021364 [0.021103, 0.021627] |
| population-s11-u14004 | 0.023756 [0.023467, 0.024034] | 0.017531 [0.017129, 0.017946] | 0.021337 [0.021075, 0.021603] |
| population-s29-u14004 | 0.023707 [0.023423, 0.023985] | 0.017552 [0.017144, 0.017972] | 0.021346 [0.021086, 0.021607] |
| population-seedmean-u14004 | 0.023731 [0.023446, 0.024008] | 0.017541 [0.017138, 0.017956] | 0.021342 [0.021081, 0.021603] |

### Fixed A-defined uncertain presence Brier

| Model/checkpoint | Unknown hand | Draw pile | Prizes |
|---|---:|---:|---:|
| A | 0.184585 [0.182561, 0.186575] | 0.178558 [0.174046, 0.183286] | 0.163908 [0.160948, 0.166887] |
| control-s11-u00000 | 0.184585 [0.182561, 0.186575] | 0.178558 [0.174046, 0.183286] | 0.163908 [0.160948, 0.166887] |
| control-s29-u00000 | 0.184585 [0.182561, 0.186575] | 0.178558 [0.174046, 0.183286] | 0.163908 [0.160948, 0.166887] |
| control-seedmean-u00000 | 0.184585 [0.182561, 0.186575] | 0.178558 [0.174046, 0.183286] | 0.163908 [0.160948, 0.166887] |
| population-s11-u00000 | 0.184585 [0.182561, 0.186575] | 0.178558 [0.174046, 0.183286] | 0.163908 [0.160948, 0.166887] |
| population-s29-u00000 | 0.184585 [0.182561, 0.186575] | 0.178558 [0.174046, 0.183286] | 0.163908 [0.160948, 0.166887] |
| population-seedmean-u00000 | 0.184585 [0.182561, 0.186575] | 0.178558 [0.174046, 0.183286] | 0.163908 [0.160948, 0.166887] |
| control-s11-u14004 | 0.179660 [0.177451, 0.181802] | 0.176541 [0.172339, 0.180904] | 0.161535 [0.158665, 0.164381] |
| control-s29-u14004 | 0.178636 [0.176466, 0.180704] | 0.176078 [0.171844, 0.180457] | 0.160937 [0.158218, 0.163679] |
| control-seedmean-u14004 | 0.179148 [0.176958, 0.181238] | 0.176309 [0.172089, 0.180701] | 0.161236 [0.158436, 0.164037] |
| population-s11-u14004 | 0.177454 [0.175232, 0.179600] | 0.174285 [0.169680, 0.179000] | 0.161255 [0.158487, 0.164040] |
| population-s29-u14004 | 0.177126 [0.174920, 0.179231] | 0.174376 [0.169822, 0.179122] | 0.161258 [0.158505, 0.164033] |
| population-seedmean-u14004 | 0.177290 [0.175098, 0.179407] | 0.174330 [0.169767, 0.179037] | 0.161256 [0.158508, 0.164046] |

### Final legality and count consistency

| Final model | Support mass violation | Hand / draw-pile / prize total MAE | Card-total violation MAE / fraction |
|---|---:|---|---|
| control-s11-u14004 | 0.000000000 | 0.334996 / 0.508864 / 0.199575 | 0.001923 / 0.035730 |
| control-s29-u14004 | 0.000000000 | 0.244023 / 0.436637 / 0.166214 | 0.001827 / 0.052364 |
| control-seedmean-u14004 | 0.000000000 | 0.289509 / 0.472750 / 0.182894 | 0.001875 / 0.044047 |
| population-s11-u14004 | 0.000000000 | 0.227309 / 0.392031 / 0.176281 | 0.002817 / 0.072935 |
| population-s29-u14004 | 0.000000000 | 0.232755 / 0.439869 / 0.151254 | 0.002519 / 0.058541 |
| population-seedmean-u14004 | 0.000000000 | 0.230032 / 0.415950 / 0.163767 | 0.002668 / 0.065738 |

Marginal count consistency is diagnostic; no projection or behavior correction was introduced. Hand targets exclude legally known hand cards; those cards are added back when checking zone totals. Fixed uncertain subsets use A presence probability in [0.1,0.9] separately per zone and are identical across models.

## Learning curves

!Frozen checkpoint learning curves (historical/private artifact omitted)

Every registered checkpoint is retained. Full metrics and intervals for all seven checkpoints, both arms and seeds are in analysis.json; primary comparisons always use update 14,004.

## Secondary selection with initialization eligible

| Run | Selected update | n/100 combined validation objective |
|---|---:|---:|
| control-s11 | 0 | 0.637600 |
| population-s11 | 4668 | 0.605435 |
| control-s29 | 0 | 0.637600 |
| population-s29 | 4668 | 0.608438 |

This secondary rule minimizes the existing combined validation objective, with exact ties favoring the earlier checkpoint. It does not replace the primary endpoint. It reuses validation for selection and description, so selected-checkpoint results are not independent confirmation; conditional bootstrap intervals do not include selection uncertainty.

## What is established and unresolved

Treatment improves fixed-final deckNLL over matched control on these fitted seeds. The matched comparison changes sampling and scalar weighting together as registered; it tests their practical implementation of the same expected objective. It does not isolate AdamW from clipping, establish universal superiority across seeds, or prove that shared-head interference is absent.

Treatment improvement over initial: not established; difference -0.009556 [-0.026571, +0.006974].

Treatment improvement over A: not established; difference -0.009556 [-0.026571, +0.006974].

Hidden-zone preservation must be read separately from deck identification. The following are final treatment minus reference seed-mean differences in n/100 (negative is better); intervals crossing zero remain inconclusive, and a tradeoff is not labelled preservation.

| Reference | Metric | Difference, 95% interval |
|---|---|---:|
| control | count_NLL | -0.000668 [-0.000929, -0.000433] |
| control | count_NLL_hand | -0.000847 [-0.001102, -0.000591] |
| control | count_NLL_draw_pile | -0.000974 [-0.001517, -0.000507] |
| control | count_NLL_prizes | -0.000183 [-0.000345, -0.000030] |
| control | presence_hand | -0.000250 [-0.000318, -0.000180] |
| control | presence_draw_pile | -0.000239 [-0.000358, -0.000133] |
| control | presence_prizes | -0.000008 [-0.000053, +0.000034] |
| control | uncertain_presence_hand | -0.001705 [-0.002195, -0.001203] |
| control | uncertain_presence_draw_pile | -0.001526 [-0.002164, -0.000882] |
| control | uncertain_presence_prizes | +0.000104 [-0.000218, +0.000422] |
| initial | count_NLL | -0.002040 [-0.002365, -0.001714] |
| initial | count_NLL_hand | -0.003017 [-0.003435, -0.002564] |
| initial | count_NLL_draw_pile | -0.001673 [-0.002159, -0.001198] |
| initial | count_NLL_prizes | -0.001431 [-0.001837, -0.001040] |
| initial | presence_hand | -0.000954 [-0.001103, -0.000808] |
| initial | presence_draw_pile | -0.000387 [-0.000506, -0.000274] |
| initial | presence_prizes | -0.000381 [-0.000488, -0.000273] |
| initial | uncertain_presence_hand | -0.007767 [-0.008892, -0.006670] |
| initial | uncertain_presence_draw_pile | -0.003975 [-0.005150, -0.002802] |
| initial | uncertain_presence_prizes | -0.002676 [-0.003479, -0.001877] |
| A | count_NLL | -0.002040 [-0.002365, -0.001713] |
| A | count_NLL_hand | -0.003017 [-0.003435, -0.002564] |
| A | count_NLL_draw_pile | -0.001673 [-0.002159, -0.001198] |
| A | count_NLL_prizes | -0.001431 [-0.001837, -0.001040] |
| A | presence_hand | -0.000954 [-0.001103, -0.000808] |
| A | presence_draw_pile | -0.000387 [-0.000506, -0.000274] |
| A | presence_prizes | -0.000381 [-0.000488, -0.000273] |
| A | uncertain_presence_hand | -0.007767 [-0.008892, -0.006670] |
| A | uncertain_presence_draw_pile | -0.003975 [-0.005150, -0.002802] |
| A | uncertain_presence_prizes | -0.002676 [-0.003479, -0.001877] |

No deployment, test confirmation or follow-up experiment is launched.

## Stationary training deck NLL

| Run/checkpoint | Intended n/100 | Passing n/82 | Uniform collection |
|---|---:|---:|---:|
| control-s11-u00000 | 0.490783 | 0.549839 | 0.771852 |
| control-s11-u14004 | 0.833571 | 0.970163 | 0.600976 |
| control-s29-u00000 | 0.490783 | 0.549839 | 0.771852 |
| control-s29-u14004 | 0.714026 | 0.826437 | 0.591942 |
| population-s11-u00000 | 0.490783 | 0.549839 | 0.771852 |
| population-s11-u14004 | 0.476675 | 0.534082 | 0.827926 |
| population-s29-u00000 | 0.490783 | 0.549839 | 0.771852 |
| population-s29-u14004 | 0.503965 | 0.560893 | 0.891568 |

These are stationary replays at initialization/final, not online losses collected while parameters changed.

## Initialization parity

| Run | Max deck probability difference from A | Max count difference from floored/masked A | Max count difference from raw A |
|---|---:|---:|---:|
| control-s11 | 0.000000060 | 0.000000179 | 0.000169992 |
| control-s29 | 0.000000060 | 0.000000179 | 0.000169992 |
| population-s11 | 0.000000060 | 0.000000179 | 0.000169992 |
| population-s29 | 0.000000060 | 0.000000179 | 0.000169992 |

Count initialization deliberately applies the historical 1e-5 floor inside legal count support, then normalizes. This quantified difference from raw A is retained, not called an error. Deck parity tolerance is 2e-7 absolute; count parity is 2e-6 absolute plus 2e-5 relative. All original model/loss tests and new sampling/control-path checks precede training.

## Resource accounting

Accounted CPU 1.847 core hours; measured execution 1.680. Preparation includes a conservative 600-second charge, not invented actual measurement. UTC elapsed 0.762 hours. Completed optimizer updates 56016/56,016. Test payload/model access, GPU, native matches, PPO, paid APIs, uploads and submissions: zero.

Evidence: registration (historical/private artifact omitted), provenance (historical/private artifact omitted), forecast (historical/private artifact omitted), fixed schedules (historical/private artifact omitted), initialization (historical/private artifact omitted), sampler checks (historical/private artifact omitted), analysis (historical/private artifact omitted), fit receipts (historical/private artifact omitted), cost receipt (historical/private artifact omitted). Private checkpoints remain outside Git.
