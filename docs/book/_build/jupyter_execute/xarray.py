# xarray
xarray is a library for labelled multidimensional array objects (dataset and dataarrays).

Tutorial based on excellent guide from [Pangeo](http://gallery.pangeo.io/repos/pangeo-data/pangeo-tutorial-gallery/xarray.html).

import xarray as xr
import numpy as np

da = xr.DataArray([1, 2, 3])
da

lat = np.arange(-60, 85, 0.25)
lon = np.arange(-180, 180, 0.25)

lat[0:10]

lon[0:10]

random_array = np.random.rand(np.shape(lat)[0], np.shape(lon)[0])

random_array

da = xr.DataArray(
    random_array, 
    dims=('lat', 'lon'),
    coords={'lat': lat, 'lon': lon}
)
da

ds = da.to_dataset(name='random_array')
ds

ds.random_array.plot();

url = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc'
ds = xr.open_dataset(url, drop_variables=['time_bnds'])
ds

sst = ds['sst']
sst

sst.sel(time='2020-01-01').plot(vmin=-2, vmax=30);

sst.sel(lon=180, lat=0).plot();

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=(10, 5))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()

sst.sel(
    time='2020-01-01'
).plot(
    ax=ax, 
    transform=ccrs.PlateCarree(), 
    vmin=2, 
    vmax=30, 
    cbar_kwargs={'shrink': 0.8}
)

plt.show()

import xesmf as xe

global_grid = xr.Dataset(
    {'lat': (['lat'], np.arange(-60, 85, 0.25)), 
     'lon': (['lon'], np.arange(-180, 180, 0.25)),}
)

sst_2021 = sst.isel(time=-1)

regridder = xe.Regridder(
    sst_2021, 
    global_grid, 
    'bilinear', 
    periodic=True # needed for global grids, otherwise miss the meridian line
)
# for multiple files to the same grid, add: reuse_weights=True

sst_2021_regridded = regridder(sst_2021)

sst_2021_regridded.plot();

For more information, see the [documentation](http://xarray.pydata.org/en/stable/).