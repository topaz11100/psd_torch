# Membrane Constant Fixed Scenarios

These configs fix threshold and membrane/filter parameters to provide conservative non-trainable controls.

`simple/` contains `deap`, `uci-har`, `s-mnist`, and `shd`. All other datasets are placed in `hard/`.

Common fixed controls:

```yaml
"v_th": ["fixed", 1.0]
```

For LIF:

```yaml
"filter": "0.5"
```

For RF:

```yaml
"filter": "0.25"
```

The RF value is a neutral normalized mid-band center frequency, not a universal literature default.
