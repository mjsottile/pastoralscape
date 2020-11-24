###########################################################################
# MIT License
#
# Copyright (c) 2020 Matthew Sottile
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###########################################################################
import numpy as np
import csv
import scipy.interpolate as interp

# {{{ GISData
class GISData:
  # {{{ constructor
  """
  Store for all GIS data over time period.  This will cache all data in
  memory so we can compute over the entire time period instead of on-demand
  loading of data for each timestep.
  """
  def __init__(self, params):
    # data will be organized as a dictionary of dictionaries:
    #
    # year -> month -> data frame
    #
    self.data = {}
    self.params = params
    self.coordinates = None
    self.waterbodies = None
    self.villages = None
    self.fci = None
    self.paths = {}

    # mean NDVI per cell
    self.mean_ndvi_alltime = {}

    # load data for static features
    self.read_coordinates()
    self.read_fci()
    self.read_fci_new(years=(params['model']['setup']['start_date'].year, params['model']['setup']['end_date'].year))
    self.interpolate_fci_average()
    self.read_static_locations()
    self.read_paths()
  # }}}

  # {{{ load
  def load(self, start_year, start_month, end_year, end_month):
    """
    load all data over specified time period.
    """
    cur_year = start_year
    cur_month = start_month

    keep_reading = True
    nmonths = 0
    while keep_reading:
      if cur_year == end_year and cur_month == end_month:
        keep_reading = False

      if cur_year not in self.data:
        self.data[cur_year] = {}
      self.data[cur_year][cur_month] = self.read_date(cur_year, cur_month)
      for cell_id in self.data[cur_year][cur_month]:
        row = self.data[cur_year][cur_month][cell_id]
        if cell_id not in self.mean_ndvi_alltime:
          self.mean_ndvi_alltime[cell_id] = 0.0
        self.mean_ndvi_alltime[cell_id] += row['mean_ndvi']

      if cur_month == 12:
        cur_month = 1
        cur_year = cur_year + 1
      else:
        cur_month = cur_month + 1
      nmonths = nmonths + 1
    
    for id in self.mean_ndvi_alltime:
      self.mean_ndvi_alltime[id] = self.mean_ndvi_alltime[id] / nmonths

  # }}}

  def csv_helper(self, fname, rowfunc):
    """
    Replacement for pandas data frame code.  Pandas CSV import
    is VERY slow and caused substantial performance degradation.
    """
    data = csv.DictReader(open(fname))
    rows = []
    fields = data.fieldnames
    for row in data.reader:
      rows.append(rowfunc(row))
    a = np.array(rows, dtype=object)
    d = {}
    for i,field in enumerate(fields):
      d[field] = a[:,i]
    return d

  # {{{ read paths
  def read_paths(self):
    with open(self.params['gis']['paths'], 'r') as file:
      reader = csv.reader(file, delimiter=',')
      for row in reader:
        origin = int(row[0])
        waypoints = [origin] + [int(point) for point in row[1].split(':')]
        if origin in self.paths:
          self.paths[origin].append(waypoints)
        else:
          self.paths[origin] = [waypoints]
  # }}}

  # {{{ read_date
  def read_date(self, y, m):
    ndvi_root   = self.params['gis']['fileroot'] + 'NDVIPolygons/'
    precip_root = self.params['gis']['fileroot'] + 'PrecipitationPolygons/'
    water_root  = self.params['gis']['fileroot'] + 'WaterPolygons/'

    ndvi_fname = ndvi_root+'CleanMeanNDVI_{0}-{1:02d}.csv'.format(y,m)
    precip_fname = precip_root+'CleanPrec_{0}-{1:02d}-01.csv'.format(y,m)
    water_fname = water_root+'CleanWater_{0}-{1:02d}-01.csv'.format(y,m)

    def rowfunc(r):
      return (int(r[0][2:]), float(r[1]))

    ndvi_data = self.csv_helper(ndvi_fname, rowfunc)
    precip_data = self.csv_helper(precip_fname, rowfunc)
    water_data = self.csv_helper(water_fname, rowfunc)

    merged = {}
    id = ndvi_data['ID']
    ndvi_mean = ndvi_data['mean']
    for i in range(len(id)):
      merged[id[i]] = {'mean_ndvi': ndvi_mean[i]}
    id = precip_data['ID']
    precip_mean = precip_data['mean']
    for i in range(len(id)):
      merged[id[i]]['mean_precip'] = precip_mean[i]
    id = water_data['ID']
    water_intersect = water_data['Intersect']
    for i in range(len(id)):
      merged[id[i]]['intersect'] = water_intersect[i]

    return merged
  # }}}

  # {{{ read_coordinates
  def read_coordinates(self):
    def rowfunc(r):
      return (int(r[0][2:]), float(r[1]), float(r[2]))
    self.coordinates = self.csv_helper(self.params['gis']['coordinates'], rowfunc)
  # }}}

  # {{{ read_static_locations
  def read_static_locations(self):
    def rowfunc(r):
      return (r[0], int(r[1]))
    self.villages = self.csv_helper(self.params['gis']['villages'], rowfunc)

    self.waterbodies = self.csv_helper(self.params['gis']['waterbodies'], 
                                       lambda r: (int(r[0]), float(r[1])))
  # }}}

  # {{{ read_fci 
  def read_fci_new(self, years=None):
    """ Read the FCI data from the file specified in the parameter list.
        If we are given a pair of years (year_lo, year_hi), calculate the
        average FCI over the time period (not including year_hi) for each
        cell in the FCI grid. """
    data = {}

    reader = csv.reader(open(self.params['gis']['fcinew'],"r"), delimiter=",")
    x = list(reader)[1:]
    
    for row in x:
      long = float(row[1])
      lat = float(row[2])
      year = int(row[3])
      fcivals = [float(i) if len(i) > 0 else 0.0 for i in row[4:]]
      if year in data:
          data[year][(lat,long)] = fcivals
      else:
          data[year] = {(lat,long): fcivals}

    if years is not None:
      self.fci_averages = {}
      (year_lo, year_hi) = years

      denom = float(year_hi-year_lo)
      # account for missing years for calculating the average
      for year in range(year_lo, year_hi):
        if year not in data:
          denom = denom - 1
      ave_mult = 1.0 / denom

      for year in range(year_lo, year_hi):
        if year in data:
          for cell in data[year]:
            months = data[year][cell]
            year_average = np.mean(months) * ave_mult
            if cell not in self.fci_averages:
              self.fci_averages[cell] = year_average
            else:
              self.fci_averages[cell] = self.fci_averages[cell] + year_average
    else:
      self.fci_averages = None

    self.fci = data

  def interpolate_fci_average(self):
    """ Map the FCI average data to the model grid as a dictionary from
        cell ID to average FCI. """
    data = self.fci_averages

    # pull out FCI cell locations and values for given year/month
    coords = []
    values = []
    for ((lat, long), value) in data.items():
        coords.append((lat, long))
        values.append(value)
    coord_array = np.array(coords)
    value_array = np.array(values)

    # interpolate FCI cells to grid cells with linear method
    grid_coords = np.array([(self.coordinates['Lat'][x],self.coordinates['Long'][x]) for x in range(len(self.coordinates['Lat']))])
    fci_interp = interp.griddata(coord_array, value_array, grid_coords, method='linear')

    self.grid_fci_averages = {}
    for i in range(len(self.coordinates['ID'])):
      self.grid_fci_averages[self.coordinates['ID'][i]] = fci_interp[i]

  def get_fci_month(self, year, month):
    """ Get the FCI data for the current month interpolated to the world grid
        and return a map from cell ID to the FCI for the month/year.  Return
        none if the year is not defined.  Assume that if a year is defined we
        have a value for all months. """
    data = self.fci

    # missing year
    if year not in data:
      return None

    # pull out FCI cell locations and values for given year/month
    coords = []
    values = []
    for ((lat, long), months) in data[year].items():
        coords.append((lat, long))
        values.append(months[month-1])
    coord_array = np.array(coords)
    value_array = np.array(values)

    # interpolate FCI cells to grid cells with linear method
    grid_coords = np.array([(self.coordinates['Lat'][x],self.coordinates['Long'][x]) for x in range(len(self.coordinates['Lat']))])
    fci_interp = interp.griddata(coord_array, value_array, grid_coords, method='linear')

    result = {}
    for i in range(len(self.coordinates['ID'])):
      result[self.coordinates['ID'][i]] = fci_interp[i]
    return result

  def read_fci(self):
    def rowfunc(r):
      if r[2] == 'NA':
        return (int(r[0]), int(r[1]), None)
      return (int(r[0]), int(r[1]), float(r[2]))
    self.fci = self.csv_helper(self.params['gis']['fci'], rowfunc)
  # }}}

  # {{{ get_date
  def get_date(self, year, month):
    return self.data[year][month]
  # }}}
# }}} 
