import os

import geopandas as gpd
import pooch
from py_wake.tests import ptf


def dk_coast(crs=None):
    files = ptf('maps/dk_coast.zip',
                known_hash='c6b90f62fddec4134762d41777dbfcc50122d8d1a0d2fa0e49a0601382aaecff',
                processor=pooch.Unzip())
    f = next(f for f in files if f.endswith(os.path.join('dk_coast', 'dk_coast.shp')))
    dk = gpd.read_file(f)
    if crs:
        dk = dk.to_crs(crs)
    return dk


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        dk = dk_coast()
        dk.plot()
        plt.show()


main()
