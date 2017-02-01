# Crustshot

This script provides functions for spatial selection, querying and processing
of crustal velocity profiles from the GSN database.

## Citing

```
Artemieva, I.M. and Mooney, W.D., Thermal thickness and evolution of Precambrian lithosphere: A global study, J. Geophys. Res., 106, 16,387 - 166, 414, 2001. 
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
