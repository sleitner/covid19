import os
import pandas as pd
from urllib.request import urlopen
import json
from abbreviate_states import abbrev_us_state, us_state_abbrev
import datetime 
import numpy as np

def _enhance_covid_stats(df):
    # assumes dates ordered and filled in ; if not filled then the active numbers will be off

    def ts_delta(grp, field):
        return grp[field].diff(periods=1)

    def ts_active(grp, field, recovery_period=28):
        return grp[field].rolling(recovery_period, min_periods=1).sum()

    def ts_preactive(grp, field, recovery_period=28):
        return grp[field].rolling(100000, min_periods=1).sum().shift(recovery_period) 

    def ts_slope(grp, field, period=3):
        # need 3 points for polyfit order 1
        return grp[field].rolling(window=period, center=True, min_periods=3)\
            .apply(lambda x: np.polyfit(np.array(range(0,len(x))), x, 1)[0], raw=True)              

    def ts_mean(grp, field, period):
        '''calculate rolling trends at the center of the period'''
        return grp[field].rolling(period, min_periods=1, center=True).mean()

    def all_lt_zero(s):
        ''' 1 for increasing, 0 for decreasing or stable'''
        return 1-int(all(i <= 0 or pd.isna(i) for i in s))

    def ts_gate(grp, field, period=14):
        '''14 day period using 3 day slopes '''
        return grp[field].rolling(window=period+3//2, center=False, min_periods=1)\
            .apply(lambda s: all_lt_zero(s), raw=True)      

    def new_daily_from_delta(df, f):
        nf = 'new_' + f
        df[nf] = df.groupby('fips', as_index=False, group_keys=False) \
                                .apply(ts_delta, field=f) 
        mindate_inds = df.groupby('fips')['date'].idxmin()
        df.loc[mindate_inds, nf] = df.loc[mindate_inds, f] 
        return df

    def calc_daily_changes(df, fields):
        for f in fields:
            df = new_daily_from_delta(df, f)
        return df

    def calc_active_cases(df):
        # cases are active if they were confirmed 28 days or less ago
        df['active'] = df.groupby('fips', as_index=False, group_keys=False) \
            .apply(ts_active, field='new_confirmed')
        return df

    df = calc_daily_changes(df, fields=('confirmed','deaths'))
    df = calc_active_cases(df)

    # recovered is everyone who is no longer active minus everyone who has died
    # recovered cannot be less than zero though 
    # (real deaths don't obey the 28-day assumption)
    df['recovered'] = df.groupby('fips', as_index=False, group_keys=False) \
                                 .apply(ts_preactive, field='new_confirmed')
    df['recovered'] = (df['recovered'] - df['deaths']).apply(lambda x: x if x >0 else 0) 

    # calculate rolling averages over various periods
    for f in ['new_confirmed']:#,'new_deaths' ]:
        for period in [3,10,]: # (3,7, 14):
            df['r'+str(period)+'_'+f] = \
                df.groupby('fips', as_index=False, group_keys=False) \
                    .apply(ts_mean, field=f, period=period)

    
    slp_period = 3
    rolling = 'r3'
    f = rolling+'_'+'new_confirmed'
    slpf = 'slope'+str(slp_period)+'_'+f
    df[slpf] = \
        df.groupby('fips', as_index=False, group_keys=False) \
            .apply(ts_slope, field=f, period=slp_period)

    slp_period = 7
    rolling = 'r10'
    f = rolling+'_'+'new_confirmed'
    slpf = 'slope'+str(slp_period)+'_'+f
    df[slpf] = \
        df.groupby('fips', as_index=False, group_keys=False) \
            .apply(ts_slope, field=f, period=slp_period)

    slpf = 'slope3_r3_new_confirmed'
    df['trend_gate'] = df.groupby('fips', as_index=False, group_keys=False).apply(ts_gate, field=slpf)
    return df

class Datasets():

    def __init__(self):
        self.date_filter = datetime.datetime.strptime('2020-03-10', '%Y-%m-%d')
        data_dir = 'data/raw/covid-19-data/'
        if not os.path.exists(data_dir):
            raise ValueError('Need NYT data to be cloned into data/raw/')
        else:
            dtypes = {'fips': str}
            self.dates_lambda = lambda x: \
                datetime.datetime.strptime(x,'%Y-%m-%d')
            self.state_lambda = lambda x: \
                us_state_abbrev[x] if x in us_state_abbrev.keys() else x

            self.county_path = data_dir + 'us-counties.csv'
            self.states_path = data_dir + 'us-states.csv'
            self.counties = pd.read_csv(self.county_path, dtype=dtypes)
            self.states = pd.read_csv(self.states_path, dtype=dtypes)
            print("Loaded county and state data")

    def process_df(self):
        self.counties.rename(columns={'cases': 'confirmed'}, inplace=True)
        dates = self.counties['date'].apply(self.dates_lambda)
        self.counties.date = dates
        logic = self.counties.date > self.date_filter
        self.counties = self.counties[logic]
        abbr = self.counties.state.apply(self.state_lambda)
        self.counties['state_abbr'] = abbr
        nyc = self.counties.county == 'New York City'
        self.counties.loc[nyc, 'fips'] = '36061'
        county_name = self.counties.county.apply(lambda x: x.title())
        county_state = self.counties.state_abbr.apply(lambda x:x.upper())
        self.counties['geo_label'] = county_name + ', ' + county_state
        self.counties = self.counties[~pd.isnull(self.counties.fips)]
        print("Processed county data")


    def covid_data(self):
        # https://github.com/nytimes/covid-19-data/
        counties = self.counties.copy()
        counties.rename(columns={'cases':'confirmed'}, inplace=True)

        # date   county  state   fips    cases   deaths
        dates = counties['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
        counties['date'] = dates
        logic = counties.date > datetime.datetime.strptime('2020-03-10','%Y-%m-%d')
        counties = counties[logic]
        counties['state_abbr'] = counties.state.apply(lambda x: us_state_abbrev[x] if x in us_state_abbrev.keys() else x)
        counties.loc[counties.county=='New York City','fips']='36061' # associate NYC with a population
        counties['geo_label'] = counties['county'].apply(lambda x: x.title()) + ', ' + counties['state_abbr'].apply(lambda x:x.upper())
        counties = counties[~pd.isnull(counties.fips)]

        fn = 'data/raw/covid-19-data/us-states.csv'
        covid_state =  pd.read_csv(fn,dtype={'fips':str,}).rename(columns={'cases':'confirmed'})
        #date   state   fips    cases   deaths
        covid_state['state_abbr'] = covid_state['state'].apply(lambda x: us_state_abbrev[x] if x in us_state_abbrev.keys() else x)
        covid_state['geo_label'] = covid_state['state'].apply(lambda x: x.title())
        covid_state = covid_state[~pd.isnull(covid_state.fips)]
        covid_state['date'] = covid_state['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
        covid_state = covid_state[covid_state.date > datetime.datetime.strptime('2020-03-10','%Y-%m-%d')]

        counties = _enhance_covid_stats(counties)
        covid_state = _enhance_covid_stats(covid_state)
        return covid_state, counties

    #population numbers
    #https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?src=bkmk#
    def population_data(self):
        geo_pop = pd.read_csv('data/raw/PEP_2018_PEPCUMCHG.ST05/PEP_2018_PEPCUMCHG.ST05_with_ann.csv', 
                         encoding='latin-1', header=1)#,dtype={"Target Geo Id2":str})
        geo_pop.rename(columns={'Population Estimate - July 1, 2018':'pop2018', 'Geography':'state'}, inplace=True)
        geo_pop['fips'] = geo_pop['Target Geo Id'].apply(lambda x: x.split('US')[-1])
        geo_pop['state_abbr'] = geo_pop['state'].apply(lambda x: us_state_abbrev[x] if x in us_state_abbrev.keys() else x)
        state_pop = geo_pop[geo_pop['fips'].apply(lambda x: len(x)==2)][['state','state_abbr','pop2018']]
        county_pop = geo_pop[geo_pop['fips'].apply(lambda x: len(x)==5)][['fips','Geography.2','state','pop2018']]
        county_pop.loc[county_pop['fips']=='36061','pop2018'] = 8.623e6 # hardcoding NYC because data looks like the wrong fips code
        # you want to match DC to county data 
        county_pop = pd.concat([county_pop, geo_pop[geo_pop['fips']=='11001'][['fips','Geography.2','state','pop2018']]])
        return state_pop, county_pop


    # geography
    def geo_data(self):
        # https://www.kaggle.com/danofer/zipcodes-county-fips-crosswalk
        # https://github.com/Data4Democracy/zip-code-to-county/blob/master/county-fips.csv
        # zip to county from HUD: https://www.huduser.gov/portal/datasets/usps_crosswalk.html
        c_zip_fips = pd.read_csv('data/processed/zip_to_fips.csv', dtype={'fips':str, 'zip':str})
        c_zip_fips.rename({'STATE':'state_abbr'})
        # https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json 
        with open('data/raw/geojson-counties-fips.json') as f:
            counties_geojson = json.load(f)
        # https://eric.clst.org/tech/usgeojson/ 
        with open('data/raw/geojson-states-fips.json') as f:
            states_geojson = json.load(f)
        #https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html
        f = 'data/raw/2019_Gaz_counties_national.txt'
        counties_latlong = pd.read_csv(f, '\t', dtype={'GEOID':str})
        counties_latlong = counties_latlong.rename(columns={'GEOID':'fips', 'NAME':'county', 'USPS':'state', 
            'INTPTLAT':'latitude', counties_latlong.columns[-1]:'longitude'})
        counties_latlong = counties_latlong[['fips', 'county', 'state', 'latitude', 'longitude']]
        # could have used: https://www.kaggle.com/washimahmed/usa-latlong-for-state-abbreviations#statelatlong.csv
        # but instead created own from above
        f = 'data/processed/state_latlong.csv'
        state_latlong = pd.read_csv(f)
        return c_zip_fips, counties_geojson, states_geojson, counties_latlong, state_latlong

