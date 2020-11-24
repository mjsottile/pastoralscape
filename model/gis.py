import pandas as pd
import numpy as np

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

    # mean NDVI per cell
    self.mean_ndvi_alltime = {}

    # load data for static features
    self.read_coordinates()
    self.read_fci()
    self.read_static_locations()
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
      
      for _,row in self.data[cur_year][cur_month].iterrows():
        if row['ID'] not in self.mean_ndvi_alltime:
          self.mean_ndvi_alltime[row['ID']] = 0.0
        self.mean_ndvi_alltime[row['ID']] += row['mean_ndvi']

      if cur_month == 12:
        cur_month = 1
        cur_year = cur_year + 1
      else:
        cur_month = cur_month + 1
      nmonths = nmonths + 1
    
    for id in self.mean_ndvi_alltime:
      self.mean_ndvi_alltime[id] = self.mean_ndvi_alltime[id] / nmonths

  # }}}

  # {{{ read_date
  def read_date(self, y, m):
    ndvi_root   = self.params['gis']['fileroot'] + 'NDVIPolygons/'
    precip_root = self.params['gis']['fileroot'] + 'PrecipitationPolygons/'
    water_root  = self.params['gis']['fileroot'] + 'WaterPolygons/'

    ndvi_df = pd.read_csv(ndvi_root+'CleanMeanNDVI_{0}-{1:02d}.csv'.format(y,m))
    precip_df = pd.read_csv(precip_root+'CleanPrec_{0}-{1:02d}-01.csv'.format(y,m))
    water_df = pd.read_csv(water_root+'CleanWater_{0}-{1:02d}-01.csv'.format(y,m))
    coordinates_df = self.coordinates

    ndvi_df.columns = ['ID', 'mean_ndvi']
    precip_df.columns = ['ID', 'mean_precip']
    water_df.columns = ['ID', 'intersect']
    
    m = pd.merge(pd.merge(ndvi_df, precip_df, on="ID", how="outer"), water_df, on="ID", how="outer")
    m['ID'] = (m.ID.str[2:]).astype(int)
    m = pd.merge(m, coordinates_df, on="ID", how="outer")
    return m.sort_values(by='ID')
  # }}}

  # {{{ read_coordinates
  def read_coordinates(self):
    coordinates_df = pd.read_csv(self.params['gis']['coordinates'])
    coordinates_df['ID'] = (coordinates_df.ID.str[2:]).astype(int)
    self.coordinates = coordinates_df
  # }}}

  # {{{ read_static_locations
  def read_static_locations(self):
    self.villages = pd.read_csv(self.params['gis']['villages'])
    self.waterbodies = pd.read_csv(self.params['gis']['waterbodies'])
  # }}}

  # {{{ read_fci 
  def read_fci(self):
    self.fci = pd.read_csv(self.params['gis']['fci'])
  # }}}

  # {{{ get_date
  def get_date(self, year, month):
    return self.data[year][month]
  # }}}
# }}} 