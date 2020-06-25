import os
import pandas as pd
from urllib.request import urlopen
import json
from abbreviate_states import abbrev_us_state, us_state_abbrev
import datetime
import numpy as np


class Calculator:

    @staticmethod
    def ts_delta(grp, field):
        return grp[field].diff(periods=1)

    @staticmethod
    def ts_preactive(grp, field, recovery_period=28):
        return grp[field]\
            .rolling(100000, min_periods=1)\
            .sum()\
            .shift(recovery_period)

    @staticmethod
    def ts_active(grp, field, recovery_period=28):
        return grp[field].rolling(recovery_period, min_periods=1).sum()

    @staticmethod
    def all_lt_zero(s):
        ''' 1 for increasing, 0 for decreasing or stable'''
        return 1 - int(all(i <= 0 or pd.isna(i) for i in s))

    @staticmethod
    def ts_slope(grp, field, period=3):
        # need 3 points for polyfit order 1
        lambda_fn = lambda x: np.polyfit(np.array(range(0, len(x))), x, 1)[0]
        return grp[field].rolling(
            window=period,
            center=True,
            min_periods=3).apply(lambda_fn, raw=True)

    @staticmethod
    def ts_mean(grp, field, period):
        '''calculate rolling trends at the center of the period'''
        return grp[field].rolling(period, min_periods=1, center=True).mean()

    def new_daily_from_delta(self, df, f):
        nf = 'new_' + f
        df[nf] = df.groupby(
            'fips',
            as_index=False,
            group_keys=False
        ).apply(self.ts_delta, field=f)
        mindate_inds = df.groupby('fips')['date'].idxmin()
        df.loc[mindate_inds, nf] = df.loc[mindate_inds, f]
        return df

    def calc_daily_changes(self, df, fields):
        for f in fields:
            df = self.new_daily_from_delta(df, f)
        return df

    def calc_active_cases(self, df):
        df['active'] = df.groupby(
            'fips',
            as_index=False,
            group_keys=False
        ).apply(self.ts_active, field='new_confirmed')
        return df

    def ts_gate(self, grp, field, period=14):
        '''14 day period using 3 day slopes '''
        return grp[field].rolling(
            window=period + 3 // 2,
            center=False,
            min_periods=1).apply(lambda s: self.all_lt_zero(s), raw=True)



def _enhance_covid_stats(df):
    calculator = Calculator()

    print("Beginning covid stats analysis...")
    df = calculator.calc_daily_changes(df, fields=('confirmed', 'deaths'))
    df = calculator.calc_active_cases(df)

    # recovered is everyone who is no longer active minus everyone who has died
    # recovered cannot be less than zero though
    # (real deaths don't obey the 28-day assumption)
    df['recovered'] = df.groupby('fips', as_index=False, group_keys=False) \
        .apply(calculator.ts_preactive, field='new_confirmed')
    diff = (df['recovered'] - df['deaths'])
    df['recovered'] = diff.apply(lambda x: x if x > 0 else 0)

    print("Calculated initial stats")
    # calculate rolling averages over various periods
    for f in ['new_confirmed']:  # ,'new_deaths' ]:
        for period in [3, 10, ]:  # (3,7, 14):
            df['r' + str(period) + '_' + f] = \
                df.groupby('fips', as_index=False, group_keys=False) \
                    .apply(calculator.ts_mean, field=f, period=period)

    print("Calculated rolling means")

    slp_period = 3
    rolling = 'r3'
    f = rolling + '_' + 'new_confirmed'
    slpf = 'slope' + str(slp_period) + '_' + f
    df[slpf] = \
        df.groupby('fips', as_index=False, group_keys=False) \
            .apply(calculator.ts_slope, field=f, period=slp_period)

    slp_period = 7
    rolling = 'r10'
    f = rolling + '_' + 'new_confirmed'
    slpf = 'slope' + str(slp_period) + '_' + f
    df[slpf] = \
        df.groupby('fips', as_index=False, group_keys=False) \
            .apply(calculator.ts_slope, field=f, period=slp_period)

    print("Completed calculating slopes")

    slpf = 'slope3_r3_new_confirmed'
    df['trend_gate'] = df.groupby(
        'fips',
        as_index=False,
        group_keys=False
    ).apply(calculator.ts_gate, field=slpf)
    print("Completed calculating enhanced statistics")
    return df


abbrevs = us_state_abbrev.keys()




class Datasets:

    @staticmethod
    def dates_lambda(x): return datetime.datetime.strptime(x, '%Y-%m-%d')

    @staticmethod
    def state_lambda(x): return us_state_abbrev[x] if x in abbrevs else x

    def __init__(self):
        self.date_filter = datetime.datetime.strptime('2020-03-10', '%Y-%m-%d')

        self.data_dir = 'data/raw/'
        self.nyt_dir = 'covid-19-data/'
        self.pop_dir = 'PEP_2018_PEPCUMCHG.ST05/'

        pop_file = 'PEP_2018_PEPCUMCHG.ST05_with_ann.csv'
        self.county_path = self.data_dir + self.nyt_dir + 'us-counties.csv'
        self.states_path = self.data_dir + self.nyt_dir + 'us-states.csv'
        self.pop_path = self.data_dir + self.pop_dir + pop_file
        self.latlon_path = self.data_dir + '2019_Gaz_counties_national.txt'

        dtypes = {'fips': str}
        self.counties = pd.read_csv(self.county_path, dtype=dtypes)
        self.states = pd.read_csv(self.states_path, dtype=dtypes)
        self.pop = pd.read_csv(self.pop_path, encoding='latin-1', header=1)
        self.states_population = None
        self.counties_population = None

        dtype = {'GEOID': str}
        self.counties_latlon = pd.read_csv(self.latlon_path, '\t', dtype=dtype)

        with open(self.data_dir + 'geojson-counties-fips.json') as f:
            self.counties_geojson = json.load(f)
        with open(self.data_dir + 'geojson-states-fips.json') as f:
            self.states_geojson = json.load(f)

        self.state_latlong = pd.read_csv('data/processed/state_latlong.csv')

        f = 'data/processed/zip_to_fips.csv'
        dtype = {'fips': str, 'zip': str}
        self.c_zip_fips = pd.read_csv(f, dtype=dtype)
        self.c_zip_fips.rename({'STATE': 'state_abbr'})

        print("Loaded raw data")

    def process_df(self, geo):
        df = None
        if geo == "counties":
            df = self.counties.copy()
        elif geo == "states":
            df = self.states.copy()

        df.rename(columns={'cases': 'confirmed'}, inplace=True)
        df.date = df['date'].apply(self.dates_lambda)
        df = df[df.date > self.date_filter]
        abbr = df.state.apply(self.state_lambda)
        df['state_abbr'] = abbr
        if geo == "counties":
            nyc = df.county == 'New York City'
            df.loc[nyc, 'fips'] = '36061'
            county_name = df.county.apply(lambda x: x.title())
            county_state = df.state_abbr.apply(lambda x: x.upper())
            df['geo_label'] = county_name + ', ' + county_state
        elif geo == "states":
            df['geo_label'] = df['state'].apply(lambda x: x.title())
        df = df[~pd.isnull(df.fips)]

        if geo == "counties":
            self.counties = df
        elif geo == "states":
            self.states = df

        print("Processed " + geo + " data")

    def covid_data(self):
        self.process_df("counties")
        self.process_df("states")
        counties = _enhance_covid_stats(self.counties)
        states = _enhance_covid_stats(self.states)
        return states, counties

    def process_population(self):
        col_rename = {
            'Population Estimate - July 1, 2018': 'pop2018',
            'Geography': 'state'
        }
        self.pop.rename(columns=col_rename, inplace=True)

        def fn(x): return x.split('US')[-1]

        ids = self.pop['Target Geo Id'].apply(fn)
        states = self.pop.state.apply(self.state_lambda)
        self.pop['fips'] = ids
        self.pop['state_abbr'] = states

        logic = self.pop['fips'].apply(lambda x: len(x) == 2)
        cols = ['state', 'state_abbr', 'pop2018']
        self.states_population = self.pop[logic][cols]

        logic = self.pop['fips'].apply(lambda x: len(x) == 5)
        cols = ['fips', 'Geography.2', 'state', 'pop2018']
        self.counties_population = self.pop[logic][cols]

        logic = self.counties_population['fips'] == '36061'
        self.counties_population.loc[logic, 'pop2018'] = 8.623e6

        logic = self.pop['fips'] == '11001'
        cols = ['fips', 'Geography.2', 'state', 'pop2018']
        temp = self.pop[logic][cols]
        self.counties_population = pd.concat([self.counties_population, temp])

        print("Processed population data")

    def population_data(self):
        self.process_population()
        return self.states_population, self.counties_population

    # geography
    def geo_data(self):
        cols = {
            'GEOID': 'fips',
            'NAME': 'county',
            'USPS': 'state',
            'INTPTLAT': 'latitude',
            self.counties_latlon.columns[-1]: 'longitude'
        }
        self.counties_latlon.rename(columns=cols, inplace=True)
        cols = ['fips', 'county', 'state', 'latitude', 'longitude']
        self.counties_latlon = self.counties_latlon[cols]

        print("Processed geography data")

        return self.c_zip_fips, \
               self.counties_geojson, \
               self.states_geojson, \
               self.counties_latlon, \
               self.state_latlong
