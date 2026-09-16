# Nutrition5k pilot manifest

The pilot is selected from the official `rgb_train_ids.txt` and
`rgb_test_ids.txt` files using seed `20260920` and SHA-256 ranking of
`seed:split:dish_id`. There is no class or café balancing. The seed is the
first value from `20260916` onward for which all 40 selected dishes occur in
the 4,793 side-angle dish directories indexed from the official bucket.

The exact 32 train and 8 test IDs below were verified against the official
files on 2026-09-16. Validate the local selection and counts with:

```bash
python3 -m nutrition5k check
```

## Train (32)

```text
dish_1550785404
dish_1551122871
dish_1551228582
dish_1551233313
dish_1551318183
dish_1551375596
dish_1551491048
dish_1551492732
dish_1551494178
dish_1551563824
dish_1551568156
dish_1558116298
dish_1558628760
dish_1558629517
dish_1558723512
dish_1558723818
dish_1559842409
dish_1560454539
dish_1561060925
dish_1561664061
dish_1561666619
dish_1562094730
dish_1562687246
dish_1562785369
dish_1562790170
dish_1562874203
dish_1562959599
dish_1564169878
dish_1564170923
dish_1565033220
dish_1566230258
dish_1568144879
```

## Test (8)

```text
dish_1550770447
dish_1551227380
dish_1551235600
dish_1551391543
dish_1558722125
dish_1562099076
dish_1562099134
dish_1565974375
```
