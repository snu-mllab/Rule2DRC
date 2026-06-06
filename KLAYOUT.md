# Installing KLayout

Rule2DRC uses KLayout to execute DRC decks. The Python dependency in `pyproject.toml` installs the `pya` package, but local evaluation also needs the `klayout` command-line binary on `PATH`.

Verify your install with:

```bash
klayout -v
```

## macOS

Download KLayout 0.30.5:

https://www.klayout.org/downloads/MacOS/HW-klayout-0.30.5-macOS-Sonoma-1-qt5MP-RsysPhb311.dmg

Then expose the binary on `PATH`:

```bash
mkdir -p ~/.local/bin
ln -s "/Applications/KLayout.app/Contents/MacOS/klayout" ~/.local/bin/klayout
export PATH="$HOME/.local/bin:$PATH"
klayout -v
```

## Linux

Build KLayout 0.30.5 from source at commit `dacb323`. The commands below place the checkout at `$HOME/klayout`:

```bash
cd "$HOME"
git clone https://github.com/KLayout/klayout.git
cd klayout
git checkout dacb323

conda create -n klaqt5 -c conda-forge python=3.10 ruby=3.2 qt-main=5.15.* qttools zlib libgit2 make cmake -y
conda activate klaqt5
conda install -n klaqt5 -c conda-forge libgit2 pkg-config

export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

./build.sh \
  -qmake  "$(which qmake)" \
  -python "$(which python)" \
  -ruby   "$(which ruby)" \
  -pyinc  "$CONDA_PREFIX/include/python3.10" \
  -pylib  "$CONDA_PREFIX/lib/libpython3.10.so.1.0" \
  -without-qtbinding \
  -j 8

echo 'export PATH="$HOME/klayout/bin-release:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$HOME/klayout/bin-release:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc

klayout -v
conda deactivate
```

If the build fails with OpenGL-related errors, install the system OpenGL/Mesa development files and re-run the build:

```bash
sudo apt-get install -y mesa-common-dev libgl1-mesa-dev
```

For headless machines, this environment variable is often needed:

```bash
export QT_QPA_PLATFORM=offscreen
```
