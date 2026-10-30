# Nutrition5k pilot manifest

The pilot is selected from the official overhead split files
`depth_train_ids.txt` and `depth_test_ids.txt` using seed `20260920` and
SHA-256 ranking of `seed:split:dish_id`. There is no class or café balancing.
The seed was chosen on 2026-09-16 for the side-angle split and kept when the
project moved to the overhead view (2026-10-30); every one of the 40 dishes
below has an overhead `rgb.png` and `depth_raw.png` upstream (checked with
`prepare` and `check` on 2026-10-30).

Validate the local selection and counts with:

```bash
python3 -m nutrition5k check
```

## Train (32)

```text
dish_1558116298
dish_1558628760
dish_1558629517
dish_1558723512
dish_1558723818
dish_1559842409
dish_1560454539
dish_1560787817
dish_1561060925
dish_1561664061
dish_1561666619
dish_1561737293
dish_1562094730
dish_1562687246
dish_1562775110
dish_1562785369
dish_1562790170
dish_1562874203
dish_1562959599
dish_1562963059
dish_1563302391
dish_1563389199
dish_1564169878
dish_1564170923
dish_1565033220
dish_1565118029
dish_1565119739
dish_1565724509
dish_1565725047
dish_1565798531
dish_1565987081
dish_1568401064
```

## Test (8)

```text
dish_1558031302
dish_1558722125
dish_1562099076
dish_1562099134
dish_1563566965
dish_1565123783
dish_1565974375
dish_1566849987
```
