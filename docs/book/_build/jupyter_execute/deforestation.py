# Deforestation tiles

## Global Forest Change 2000-2018 v1.6 data
- Download [data](https://earthenginepartners.appspot.com/science-2013-global-forest/download_v1.6.html).  
- First find the cumulative loss over all years.  
- Combine mosaic tiles.  
- Regrid.  
- Find the individual loss for each year.  

import numpy as np
import gdal
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
from matplotlib import colors

path = '/nfs/a68/earlacoa/deforestation/hansen_gfs_v1.6_lossyear/'

def regrid_mosaic_defor(res, year):
    """
    Description:
        The Hansen dataset provides the year of forest loss per grid cell between 2000 and 2018.
        For each year (e.g. 2005), convert the forest loss to binary by converting all forest loss up to that year
        (e.g. 2000-2005) to a 1 and all forest loss in future years to a 0 (e.g. 2006-2018).
        This is the cumulative forest loss.
        Then to obtain individual forest loss for each year after 2000, subtract the forest loss from the
        previous year (e.g. individual 2006 = cumulative 2006 - cumulative 2005).
    Args:
        res (float): desired resolution in degrees.
        year (int): year.
    Returns:
        hdat (2d array, float): regridded deforestation data.
        lat (1d array, float): latitude.
        lon (1d array, float): longitude.
    """
    # setup domain
    lon_extent_min = -180
    lon_extent_max = 180
    lat_extent_min = -60
    lat_extent_max = 80
    scale = int(res / abs(0.00025)) # fixed
    ydim = (int((lat_extent_max - lat_extent_min) / res))
    xdim1 = (int(10 / res)) # fixed
    xdim2 = (int((lon_extent_max - lon_extent_min) / res))
    vdat = np.empty((ydim, xdim1))
    hdat = np.empty((ydim, xdim2))
    fname = ''
    j = 0
    
    for nx in np.arange(-lon_extent_min, -lon_extent_max, -10):
        i = 0
        
        for ny in np.arange(lat_extent_max, lat_extent_min, -10):
            # define the filename based on lat and lon as uses north, south, east, and west in filenames
            if (nx > 0 or nx == 180) and ny >= 0:
                xstr = str(nx).zfill(3)
                ystr = str(ny).zfill(2)
                fname = 'Hansen_GFC-2018-v1.6_lossyear_' + ystr + 'N_' + xstr + 'W.tif'
            elif (nx > 0 or nx == 180) and ny < 0:
                xstr = str(nx).zfill(3)
                ystr = str(abs(ny)).zfill(2)
                fname = 'Hansen_GFC-2018-v1.6_lossyear_' + ystr + 'S_' + xstr + 'W.tif'
            elif nx <= 0 and ny >= 0:
                xstr = str(abs(nx)).zfill(3)
                ystr = str(ny).zfill(2)
                fname = 'Hansen_GFC-2018-v1.6_lossyear_' + ystr + 'N_' + xstr + 'E.tif' 
            elif nx <= 0 and ny < 0:
                ystr = str(abs(ny)).zfill(2)
                xstr = str(abs(nx)).zfill(3)
                fname = 'Hansen_GFC-2018-v1.6_lossyear_' + ystr + 'S_' + xstr + 'E.tif'
            print(fname)
            
            # read data
            ds = gdal.Open(path + fname)
            band = ds.GetRasterBand(1)
            loss_data = band.ReadAsArray()
            
            # create new lat and lon
            gt  = ds.GetGeoTransform()
            lon = np.linspace(gt[0], gt[0] + gt[1] * loss_data.shape[1], loss_data.shape[1])
            lat = np.linspace(gt[3], gt[3] + gt[5] * loss_data.shape[0], loss_data.shape[0])
            xx, yy = lon[::scale], lat[::scale] 
            
            # transform to binary for the year of interest
            loss_data[loss_data > (year - 2000)] = 0
            loss_data[(loss_data >= 1) & (loss_data <= (year - 2000))] = 1
            
            # equate coarse grid to high-res grid
            data_regrid = np.zeros((xdim1, xdim1))
            
            for ix in range(len(xx)):
                temp_lon_ix = np.where((lon == xx[ix]))[0]
                
                for iy in range(len(yy)):                   
                    temp_lat_iy = np.where((lat == yy[iy]))[0]
                    
                    # trim the data
                    data_trim = (loss_data[temp_lat_iy[0]:temp_lat_iy[0] + scale, :][:, temp_lon_ix[0]:temp_lon_ix[0] + scale])
                    
                    # find average across this trimmed data
                    mean_loss = np.nanmean(data_trim)
                    
                    # set this mean to the corresponding value on the coarser grid
                    data_regrid[iy, ix] = mean_loss
                    
            # add each tiles' regridded data to a new global data
            vdat[i * xdim1:(i + 1) * xdim1, :] = data_regrid
            i += 1
            
        hdat[:, j * xdim1:(j + 1) * xdim1] = vdat
        j += 1
        
    # create final global lat and lon
    lat = np.arange(lat_extent_max, lat_extent_min, -res)
    lon = np.arange(lon_extent_min, lon_extent_max, res)
    
    return hdat, lat, lon

def plot(defor_xx, defor_yy, defor_array, year, values_max, label):
    """
    Description:
        Create a contoured plot of the global deforestation.
        Either cumulatively or individually per year.
    Args:
        defor_xx (2d array, float): longitude.
        defor_yy (2d array, float): latitude.
        defor_array (2d array, float): deforestation.
        year (int): year.
        values_max (float): maximum value for the colour bar.
        label (str): cumulative or individual.
    Returns:
        Plot displayed and saved to file.
    """
    fig = plt.figure(1, figsize=(14, 7))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])
    basemap = Basemap(
        llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90,
        projection='cyl', resolution='l'
    )
    basemap.drawcountries(linewidth=0.2)
    basemap.drawcoastlines(linewidth=0.2)
    basemap.fillcontinents(color='lightgrey', zorder=0)
    basemap.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=14, linewidth=0)
    basemap.drawmeridians(np.arange(-180., 181., 30.), labels=[0, 0, 0, 1], fontsize=14, linewidth=0)
    cmap = 'viridis'
    norm = colors.Normalize(vmin=0, vmax=values_max)
    im = basemap.contourf(
        defor_xx, defor_yy, defor_array, 
        np.linspace(0, values_max, 11), cmap=cmap, norm=norm
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=im.cmap)
    sm.set_array([])
    cb = basemap.colorbar(sm, "right", size="5%", pad='2%', norm=norm, cmap=cmap, ticks=im.levels)
    cb.set_label(
        'Hansen GFC 2018 v1.6, loss year, 0.25 degrees,\n fractional deforestation (0-1)', 
        size=14
    )
    cb.ax.tick_params(labelsize=14)
    plt.title(str(year), size=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.savefig(
        path + 'Hansen_GFC-2018-v1.6_lossyear-' + label + '_global_0.25deg_' + str(year) + '.png', 
        dpi=200, alpha=True, bbox_inches='tight'
    )
    plt.show()

years = np.linspace(2000, 2018, 19, dtype=int)
res = 0.25
defor_cumulative = {}
defor_individual = {}

for year in years:
    # run regridding and mosaicing function
    output, lat, lon = regrid_mosaic_defor(res, year)

    # create dataarray
    defor = xr.DataArray(
        data = output, 
        coords = [lat, lon],
        dims = ['lat', 'lon']
    )
    defor.name = 'deforestation'
    defor = defor.assign_coords({'time': datetime.strptime(str(year)[-2:], '%y')})
    defor = defor.expand_dims('time')
    defor.to_netcdf(path + 'Hansen_GFC-2018-v1.6_lossyear-cumulative_global_0.25deg_' + str(year) + '.nc')

    defor_array = defor.isel(time=0).values
    defor_xx, defor_yy = np.meshgrid(defor.lon.values, defor.lat.values)
    
    # plot cumulative
    plot(defor_xx, defor_yy, defor_array, year, 1, 'cumulative')
    
    # calculate individual yearly loss
    defor_cumulative.update({year: xr.open_dataset(
        path + 'Hansen_GFC-2018-v1.6_lossyear-cumulative_global_' + str(res) + 'deg_' + str(year) + '.nc'
    )['deforestation']})
    if year == 2000:
        pass
    else:
        defor_individual.update({
            year: defor_cumulative[year] - defor_cumulative[year - 1]
        })
    
    defor_individual[year].to_netcdf(
        path + 'Hansen_GFC-2018-v1.6_lossyear-individual_global_' + str(res) + 'deg_' + str(year) + '.nc'
    )
    
    # plot individual
    defor_array = defor_individual[year].isel(time=0).values
    plot(defor_xx, defor_yy, defor_array, year, 0.5, 'individual')

