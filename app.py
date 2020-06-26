import copy
import datetime
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from datasets import Datasets
from ui_components import slider_choose_dates, covid_slider_mark_dates, \
    covid_selectors, caveat_markdown_text_covid

# Data sets
print("--------------------")
dat = Datasets() 
covid_state, covid_county = dat.covid_data()
state_pop, county_pop = dat.population_data()
dat.geo_data()
c_zip_fips = dat.c_zip_fips
counties_geojson = dat.counties_geojson
states_geojson = dat.states_geojson
county_latlong = dat.counties_latlon
state_latlong = dat.state_latlong
print("Completed loading datasets and computing rolled statistics")
print("--------------------")

# merge population
covid_state = pd.merge(covid_state, state_pop, on='state', suffixes=('','_'), how='inner')
covid_county = pd.merge(covid_county, county_pop, on='fips', suffixes=('','_'), how='left') #, how='inner')
# merge geography
if 'latitude' not in covid_county.columns:
    covid_county = pd.merge(
        covid_county, county_latlong[['fips', 'latitude', 'longitude']],
        on='fips', suffixes=('','_'), how='inner'
    ) 

if 'latitude' not in covid_state.columns:
    covid_state = pd.merge(
        covid_state, state_latlong,
        on='state_abbr', suffixes=('','_'), how='inner' 
    )


# Setup COVID UI
col_options, dimensions = covid_selectors()
# Setupd COVID slider
end = covid_county.date.max()
start = covid_county.date.min()
covid_period_length = (end-start).days
covid_choose_date = slider_choose_dates(start, covid_period_length+1)
covid_date_marks = covid_slider_mark_dates(covid_choose_date)


# precalculate the normalized data
def precalc_percapita(df, stat_flag):
    norm_flag = 'per 100,000'
    cfield = stat_flag + norm_flag
    if stat_flag == 'trend_gate':
        df[cfield] = df[stat_flag]
    else:
        df[cfield] = df[stat_flag] * 100000 / df['pop2018']
    return df
for covid_stat_flag_d in col_options['COVID-19 Statistic']:
    stat_flag = covid_stat_flag_d['value']
    covid_county = precalc_percapita(covid_county, stat_flag)
    covid_state = precalc_percapita(covid_state, stat_flag)


# calculate color and size scales for colorbars =====================================
covid_circle_scale = {}
covid_color_scale = {}
for covid_stat_flag_d in col_options['COVID-19 Statistic']: # see ui_components for full list
    stat_flag = covid_stat_flag_d['value']
    for covid_norm_flag_d in col_options['COVID-19 Normalization']: # '' or per_capita
        norm_flag = covid_norm_flag_d['value']
        cfield = stat_flag + norm_flag
        #calculate scale over all time
        for covid_geo_flag_d in col_options['COVID-19 Geography']: # state or county
            geo_flag = covid_geo_flag_d['value']
            if geo_flag == 'county':
                df = covid_county
            elif geo_flag == 'state':
                df = covid_state
            else:
                raise ValueError('bad geo_flag', geo_flag)
            covid_circle_scale[(geo_flag,cfield,'max')] = df[cfield].max()
            covid_circle_scale[(geo_flag,cfield,'min')] = max(
                df[cfield].min(), covid_circle_scale[(geo_flag,cfield, 'max')] * 1e-6
            )


            # overwrite some of the statflag ranges
            # trendgate is 0 or 1
            if stat_flag == 'trend_gate':
                covid_color_scale[(geo_flag, cfield, 'tvals')] = [0,1]
                covid_color_scale[(geo_flag, cfield, 'ttext')] = ['0','1']

            else:
                cmx = df[df.date >= datetime.datetime.today() - datetime.timedelta(days=30)][cfield].quantile(.95)          
                cmn = df[df.date >= datetime.datetime.today() - datetime.timedelta(days=30)][cfield].quantile(.05)

                # make slope symmetric around zero (so red is bad, blue is good)
                if stat_flag == 'slope7_r10_new_confirmed':
                    val = max(abs(cmn), abs(cmx))
                    covid_color_scale[(geo_flag, cfield, 'min')]= -val
                    covid_color_scale[(geo_flag, cfield, 'max')]= val

                ntics = 5
                val_delta =  (cmx - cmn)
                tvals = [ float('%.2g' %     (val_delta * i / (1.0 * ntics) + cmn)) for i in range(0, ntics+1) ]
                ttext = [ '%.2g' % x for x in tvals]
                ttext[-1] = '>' + ttext[-1]
                ttext[0] = '<' + ttext[0]
                covid_color_scale[(geo_flag, cfield, 'tvals')] = tvals
                covid_color_scale[(geo_flag, cfield, 'ttext')] = ttext


def choose_chloropleth(webgl_support_flag):
    if webgl_support_flag=='enabled': 
        return go.Choroplethmapbox
    else:
        return go.Choropleth
def choose_scatter(webgl_support_flag):
    if webgl_support_flag=='enabled': 
        return go.Scattermapbox
    else:
        return go.Scattergeo

app = dash.Dash(
    __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css", 'https://codepen.io/chriddyp/pen/brPBPO.css']
)

app.title = 'Alex\'s COVID-19 Dashboard'

app.layout = html.Div([
    html.Div(id='webgl-support-input', children='static-input', style=dict(visibility="hidden", display='None')),
    html.Div(id='webgl-support-output', style=dict(visibility="hidden", display='None')),
    dcc.Tabs([



        dcc.Tab(label='COVID-19 Status', children=[
        #    html.H1(children="COVID-19 Status "),
            html.Div([ # control and map section are wrapped together for flexible row
                # control section,
                html.Div(
                    [
                        html.P([
                            d['label'] + ":", 
                            dcc.Dropdown(
                                id=d['label']+'c', 
                                options=col_options[d['label']], 
                                value=d['value']
                            )
                        ]) for d in dimensions
                    ]
                    +[html.Div(id='title-covid-date-slider'+'c')]
                    +[
                        dcc.Slider(
                            id = "covid-date-slider"+'c',
                            marks = covid_date_marks, #{i: "{}".format(i) for i in [10, 20, 30, 40]},
                            min = 0,
                            max = covid_period_length,
                            step = 1,
                            value = covid_period_length
                         )
                    ]
                , className="pretty_container three columns"),
                html.Div(
                    [
                        dcc.Graph(id='output-covid-map'),
                    ], 
                className="pretty_container nine columns"),
            ],className="row flex-display"),
            html.Div([ # wrap graphs together
                html.Div(
                    [
                        dcc.Graph(id="selected-graph"),
                    ],                
                className="pretty_container six columns"),
                html.Div(
                    [
                        dcc.Graph(id="selected-graph-cum"),
                    ],                
                className="pretty_container six columns"),
            ],className="row flex-display"),
             dcc.Markdown(children=caveat_markdown_text_covid()),
        ]), #tab 2 end


        dcc.Tab(label='internal', children=[
            html.H1(children="internal data"),
        ]), #tab 1 end


    ]) #tab end
]) #div end










app.clientside_callback(
"""
function( dummy ){
  return detectWebGL()
}
""",
    Output('webgl-support-output', 'children'),
    [Input('webgl-support-input', 'children')]
)


@app.callback(
    Output('output-covid-map', 'figure'), 
    [Input(d['label']+'c', 'value') for d in dimensions] +
        [Input('covid-date-slider'+'c', 'value')] +
        [Input('webgl-support-output', 'children')]
    )
def make_covid_gates(covid_geo_flag, covid_stat_flag, covid_normalization_flag, covid_date_selected, webgl_support_flag):

    # stop if relevant variables are not present in the callback
    # this happens on load and also can occur due to communication errors 
    args = [covid_geo_flag, covid_stat_flag, covid_normalization_flag,  webgl_support_flag]
    if None in args:
        raise PreventUpdate

    fig = go.Figure()



    #  COVID overlay ============================
    cfield = covid_stat_flag + covid_normalization_flag

    if covid_geo_flag=='state':
        covid = covid_state.copy()
        covid['fips'] = covid['state_abbr']
        geojson = states_geojson
    elif covid_geo_flag=='county':
        covid = covid_county.copy()
        geojson = counties_geojson
    else:
        raise ValueError('covid_geo_flag must be "state" or "county": ', covid_geo_flag)

    covid = covid[covid.date == (covid_choose_date.loc[covid_date_selected, 'dates'].strftime('%Y-%m-%d'))]

    colorscale='Bluered',# Bluered',#'Inferno',#"Viridis",



    def covid_text_labels(covid, cfield, covid_stat_flag, covid_normalization_flag):
        text_stat = covid_stat_flag.title() + ': ' + (round(covid[cfield],2)).astype(str)
        text_norm = ' ' + covid_normalization_flag + '<br>'
        text = text_stat + ' ' + text_norm +\
            'Location: ' + covid['geo_label']+ '<br>' +\
            'Population: ' + covid['pop2018'].apply(lambda x: str(x) if pd.isna(x) else format(int(x),',d')) 
        return text

    covid['text'] = covid_text_labels(covid, cfield, covid_stat_flag, covid_normalization_flag)


    if covid_stat_flag == 'recovered':
        colorscale = 'Bluered_r'
    else:
        colorscale = 'Bluered'


    chloropleth_map = choose_chloropleth(webgl_support_flag)
    # sharing data snl https://dash.plotly.com/sharing-data-between-callbacks
    fig.add_trace(
        chloropleth_map(
            z=covid[cfield], locations=covid['fips'],
            geojson=geojson,  
            zmin = covid_color_scale[(covid_geo_flag,cfield,'tvals')][0],
            zmax = covid_color_scale[(covid_geo_flag,cfield,'tvals')][-1],
            colorscale=colorscale,# Bluered',#'Inferno',#"Viridis",
            text = covid.text,
            hovertemplate='%{text}',
            marker_opacity=0.7,
            colorbar=dict(
                tickvals=covid_color_scale[(covid_geo_flag,cfield,'tvals')],
                ticktext=covid_color_scale[(covid_geo_flag,cfield,'ttext')],
                thickness=20, 
                ticklen=3,
                title='statistic <br>'+covid_normalization_flag,
            )

        )              
    )
    if webgl_support_flag=='enabled':
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=3.,
            mapbox_center={"lat": 38.2, "lon": -96.7129},
        )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(
        title = dict(text='COVID-19 statistics',x=.5,y=0.99),
        geo_scope='usa', # limite map scope to USA
        showlegend = True,
        autosize=True,
        uirevision=True,
        clickmode= 'event+select',
        legend = dict(
        title='COVID-19 '+covid_stat_flag+' '+covid_normalization_flag,
            x=0.01, y=.01
        ),
    )  
    return fig



layout_graph = dict(
    plot_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h", x=-.1, y=-.2),
)

# Main graph -> individual graph
#@app.callback(Output("individual_graph", "figure"), [Input("main_graph", "hoverData")])
@app.callback(
    Output('selected-graph', 'figure'),
    [Input('output-covid-map', 'selectedData')] +
        [Input(d['label']+'c', 'value') for d in dimensions]
        
)
#def make_individual_figure(main_graph_hover):
def graph_selected(selected, covid_geo_flag, covid_stat_flag, covid_normalization_flag): 
    if covid_geo_flag=='state':
        covid = covid_state.copy()
        covid['fips'] = covid['state_abbr']
    elif covid_geo_flag=='county':
        covid = covid_county.copy()
    else:
        #['new_confirmed','new_deaths',   'active', 'recovered',   'confirmed', 'deaths']
        raise ValueError('covid_geo_flag must be "state" or "county": ', covid_geo_flag)

    if selected is None:
        #print('full map')
        # for full, save tinme by just aggregating state data
        df = covid_state.groupby('date', as_index=False, group_keys=False).sum()
        selected_fips = 'all' 
    else:
        #print(selected)
        selected_fips = [i['location'] for i in selected['points']]
        covid.set_index('fips', inplace=True)
        try:
            df = covid.loc[selected_fips,].groupby('date', as_index=False, group_keys=False).sum()
        except KeyError:
            print(selected_fips)
            raise PreventUpdate
        except:
            raise

    fig = make_subplots(specs=[[{"secondary_y": True}]])
#    fig = go.Figure()
    # Add traces
    fig.add_trace(
        go.Scatter(x=df.date, y=df.new_confirmed, name="new cases", 
            mode="lines+markers", line=dict(width=2, color='#a9bb95'))
    )
    fig.add_trace(
        go.Scatter(x=df.date, y=df.r10_new_confirmed, name="cases 10-day rolling average",  
            mode="lines", line=dict(width=4, color='#a9bb95', dash='dot')) #
    )
    fig.add_trace(
        go.Scatter(x=df.date, y=df.new_deaths, name="new known deaths", 
            mode="lines+markers", line=dict(width=3, color='firebrick')),
        secondary_y=True,
    )

    # Add figure title
    layout_individual = copy.deepcopy(layout_graph)
    layout_individual['title_text'] = 'Daily status in selected geography'
    fig.update_layout(layout_individual)

    # Set x-axis title
    # fig.update_xaxes(title_text="date")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>daily cases</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>daily deaths</b>", secondary_y=True)
    return fig



@app.callback(
    Output('selected-graph-cum', 'figure'),
    [Input('output-covid-map', 'selectedData')] +
        [Input(d['label']+'c', 'value') for d in dimensions]
        
)
#def make_individual_figure(main_graph_hover):
def graph_selected(selected, covid_geo_flag, covid_stat_flag, covid_normalization_flag): 

    if covid_geo_flag=='state':
        covid = covid_state.copy()
        covid['fips'] = covid['state_abbr']
    elif covid_geo_flag=='county':
        covid = covid_county.copy()
    else:
        raise ValueError('covid_geo_flag must be "state" or "county": ', covid_geo_flag)

    if selected is None:
        #print('full map')
        # for full, save tinme by just aggregating state data
        df = covid_state.groupby('date', as_index=False, group_keys=False).sum()
        selected_fips = 'all' 
    else:
        #print(selected)
        selected_fips = [i['location'] for i in selected['points']]
        covid.set_index('fips', inplace=True)
        try:
            df = covid.loc[selected_fips,].groupby('date', as_index=False, group_keys=False).sum()
        except KeyError:
            raise PreventUpdate
        except:
            raise

    fig = make_subplots(specs=[[{"secondary_y": True}]])
#    fig = go.Figure()
    # Add traces
    fig.add_trace(
        go.Scatter(x=df.date, y=df.confirmed, name="confirmed cases", 
            mode="lines+markers", line=dict(width=3, color='#a9bb95'))
    )
    fig.add_trace(
        go.Scatter(x=df.date, y=df.active, name="active cases", 
            mode="lines+markers", line=dict(width=3, color='rgba(27,158,119, .99)'))#, marker=dict(symbol="diamond-open"))
    )
    fig.add_trace(
        go.Scatter(x=df.date, y=df.deaths, name="known deaths", 
            mode="lines+markers", line=dict(width=3, color='firebrick')),
        secondary_y=True,
    )

    # Add figure title
    layout_individual = copy.deepcopy(layout_graph)
    layout_individual['title_text'] = 'Cumulative  in selected geography'
    fig.update_layout(layout_individual)

    # Set x-axis title
#    fig.update_xaxes(title_text="date")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>total cases</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>total deaths</b>", secondary_y=True)


    return fig


#====================





# Flask app for Gunicorn
server = app.server

port = os.getenv('PORT0', default=8050)
# Entrypoint for development
if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port=port)
