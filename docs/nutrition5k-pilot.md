# Nutrition5k pilot manifest

The pilot is selected from the official `rgb_train_ids.txt` and
`rgb_test_ids.txt` files using seed `20260916` and SHA-256 ranking of
`seed:split:dish_id`. There is no class or café balancing.

The exact 32 train and 8 test IDs below were verified against the official
files on 2026-09-16. Regenerate and inspect this manifest with:

```bash
python3 -m nutrition5k smoke --root data/nutrition5k --mode pilot
```

## Train (32)

```text
dish_1550781317
dish_1551317350
dish_1551378018
dish_1551393113
dish_1551396678
dish_1557862384
dish_1557863104
dish_1558380527
dish_1558629878
dish_1558724959
dish_1559059924
dish_1559235690
dish_1559245848
dish_1559590031
dish_1559845046
dish_1560543605
dish_1561404438
dish_1561576954
dish_1562614121
dish_1562686577
dish_1562688552
dish_1562789328
dish_1562961609
dish_1563391453
dish_1563996596
dish_1564429184
dish_1565207631
dish_1565379868
dish_1565383128
dish_1565809033
dish_1566413445
dish_1568649387
```

## Test (8)

```text
dish_1550773995
dish_1550777025
dish_1557937758
dish_1558376801
dish_1559157777
dish_1561662842
dish_1563551105
dish_1566590056
```
