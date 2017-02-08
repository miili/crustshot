# Crustshot

This script provides functions for spatial selection, querying and processing
of crustal velocity profiles from the GSN database.

## Citing

```
 W.D. Mooney, G. Laske and G. Masters, CRUST 5.1: A global crustal model at 5°x5°. J. Geophys. Res., 103, 727-747, 1998.
```

## Installation

Installation for different platforms:

### Linux

```
cd crustshot
sudo python setup.py install
```

or 

```
cd crustshot
pip install .
```

### Anaconda

```
cd crustshot
conda install .
```

### Installing Matplotlib.Basemap

The mapping features rely on `mpl_toolkit.basemap`. Please follow http://matplotlib.org/basemap/users/installing.html for installation hints

## Examples

Simplest example:

```
from crustshot import CrustDB

db = CrustDB()
db.plotHistogram()
```

for more see file in `examples/`
