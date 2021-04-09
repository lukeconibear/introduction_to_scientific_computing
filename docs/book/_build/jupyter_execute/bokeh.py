# Bokeh
Bokeh is a library for interactive plots.

## Simple line plot

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()

x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

plot = figure(title="Simple line example", x_axis_label="x", y_axis_label="y")
plot.line(x, y, legend_label="Temp.", line_width=2)

show(plot)

## Simple map

from bokeh.models import LogColorMapper
from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure
from bokeh.sampledata.unemployment import data as unemployment
from bokeh.sampledata.us_counties import data as counties
output_notebook()

palette = tuple(reversed(palette))
color_mapper = LogColorMapper(palette=palette)

counties = {code: county for code, county in counties.items() if county["state"] == "tx"}
county_xs = [county["lons"] for county in counties.values()]
county_ys = [county["lats"] for county in counties.values()]
county_names = [county['name'] for county in counties.values()]
county_rates = [unemployment[county_id] for county_id in counties]
data=dict(x=county_xs, y=county_ys, name=county_names, rate=county_rates)

plot = figure(
    title="Texas Unemployment, 2009", tools="pan,wheel_zoom,reset,hover,save",
    x_axis_location=None, y_axis_location=None,
    tooltips=[("Name", "@name"), ("Unemployment rate", "@rate%"), ("(Long, Lat)", "($x, $y)")])
plot.grid.grid_line_color = None
plot.hover.point_policy = "follow_mouse"
plot.patches('x', 'y', source=data,
          fill_color={'field': 'rate', 'transform': color_mapper},
          fill_alpha=0.7, line_color="white", line_width=0.5)

show(plot)

## Data source with dropdown selection

import numpy as np
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Select
from bokeh.plotting import figure
output_notebook()

x = np.linspace(0, 10, 100)
foo = x**2
bar = np.sin(x)

source = ColumnDataSource(data=dict(x=x, y=foo, foo=foo, bar=bar))

plot = figure(plot_height=350)
r = plot.line(x='x', y='foo', source=source)

select = Select(value='foo', options=['foo', 'bar'])
callback = CustomJS(args=dict(r=r, select=select), code="""
    // tell the glyph which field of the source y should refer to
    r.glyph.y.field = select.value

    // manually trigger change event to re-render
    r.glyph.change.emit()
""")
select.js_on_change('value', callback)

show(column(select, plot))

