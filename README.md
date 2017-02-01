# Crustshot

This script provides functions for spatial selection, querying and processing
of crustal velocity profiles from the GSN database

## Installation

Installation for different platforms

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

## Examples

Simplest example:

```
from crustshot import CrustDB

db = CrustDB()
db.plotHistogram()
```

for more see directory `examples/`