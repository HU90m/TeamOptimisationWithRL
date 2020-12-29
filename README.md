# DAIS-ITA Work


## Using the rustylandscapes

The rustylandscapes module is a reimplementation of the NK Landscape generator
rewritten in rust for improved speed. To build it's wheel file
(the python package format) run install `maturin`.
This can be done with the following command.
```bash
pip install manturin
```

Then to build go into the rustylandscapes directory and run manturin.
```bash
cd rustylandscapes
maturin build
```

The resulting wheel file will be shown in the last line of matuin's output.
To install this run the following.
```bash
pip install --user path/to/wheel.whl
```
