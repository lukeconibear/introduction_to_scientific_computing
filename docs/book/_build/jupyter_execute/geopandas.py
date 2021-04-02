# GeoPandas

GeoPandas is a library that extends Pandas for geospatial data.

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

df_cities = pd.DataFrame({
    'city': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
     'country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],
     'lat': [-34.58, -15.78, -33.45, 4.60, 10.48],
     'lon': [-58.66, -47.91, -70.66, -74.08, -66.86]
})
df_cities

gdf_cities = gpd.GeoDataFrame(
    df_cities,
    geometry=gpd.points_from_xy(df_cities.lon, df_cities.lat),
    crs=4326
)
gdf_cities

gdf_world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf_world.head()

gdf_south_america = gdf_world.loc[world.continent == 'South America']
gdf_south_america.head()

ax = gdf_south_america.plot(color='white', edgecolor='black')

gdf_cities.plot(ax=ax, color='green')

plt.show()

For more information, see the [documentation](https://geopandas.org/).