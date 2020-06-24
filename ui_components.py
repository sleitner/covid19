
import pandas as pd
import datetime 

def slider_choose_dates(start, period_length):
    choose_date = pd.DataFrame()
    choose_date['dates'] = pd.Series(start + datetime.timedelta(days=x) for x in range(0, period_length))
    choose_date['label'] = choose_date['dates'].apply(lambda x: x.strftime("%m/%d/%Y"))
    choose_date.index = pd.Series(range(0, period_length))
    return choose_date


# Date markers for the SES selector
def covid_slider_mark_dates(choose_date):
    date_marks = dict(zip(choose_date.index, choose_date.label))
    def reduce_mark_density(marks, sep=7):
        # start from the end and count back also add label
        return {i:{'label':marks[i]} for i in range(len(marks)-1,0,-sep)}
    date_marks = reduce_mark_density(date_marks)
    for i in date_marks:
        date_marks[i]['style'] = {
        #    'transform': 'rotate(0deg)','transform-origin': '50px',
            'writing-mode': 'vertical-rl', 
            'white-space': 'nowrap',
        }
        #    exam_date_marks[i]['tooltip'] = {'always_visible':True}
    return date_marks

# Date markers for the SES selector
def ses_slider_mark_dates(ses, start):
    exam_date_marks = set(zip(
            ses.date_index_start, 
            ses.start_date.apply(lambda x: x.strftime("%m/%d/%Y"))))|\
        (set(zip(ses.date_index_end, 
            ses.end_date.apply(lambda x: x.strftime("%m/%d/%Y")))))
    exam_date_marks = { i:j for i,j in exam_date_marks if i>=0}
    exam_date_marks[0] = start.strftime("%m/%d/%Y")
    def reduce_mark_density(marks, sep=10):
        'take a dictionary indexed by separation and return sparser dictionary'
        # also add 'label' key 
        new_mark = {}
        prev = -99
        for i,v in pd.Series(marks).sort_index().iteritems():
            if i-prev>sep:
                new_mark[i] = {'label':v}
                prev = i
            else: 
                new_mark[i] = {'label':''}
        return new_mark
    exam_date_marks = reduce_mark_density(exam_date_marks)
    for i in exam_date_marks:
        exam_date_marks[i]['style'] = {
        #    'transform': 'rotate(0deg)','transform-origin': '50px',
            'writing-mode': 'vertical-rl', 
            'white-space': 'nowrap',
        }
        #    exam_date_marks[i]['tooltip'] = {'always_visible':True}
    return exam_date_marks

def covid_selectors():
    col_options = {}
    ui_geo_label = 'COVID-19 Geography'
    dimensions = [dict(label=ui_geo_label, value='state'),]
    col_options[ui_geo_label] = [
        dict(label='State', value='state'),
        dict(label='County', value='county'),
    ]


    ui_stat_label = 'COVID-19 Statistic'
    dimensions.append(dict(label=ui_stat_label, value='confirmed'))
    col_options['COVID-19 Statistic'] = [
       dict(label='Confirmed', value='confirmed'),
       dict(label='Deaths', value='deaths'),
       dict(label='New confirmed (daily)', value='new_confirmed'),
       dict(label='New deaths (daily)', value='new_deaths'),
       dict(label='Active', value='active'),
       dict(label='Recovered', value='recovered'),
       # ----------- only give options through 0:5
       dict(label='7-day trend', value='slope7_r10_new_confirmed'),
       dict(label='CDC gate criteria (3-day trends)', value='trend_gate'),
    ]
    ui_norm_label = 'COVID-19 Normalization'
    dimensions.append(dict(label=ui_norm_label, value='per 100,000'))
    col_options[ui_norm_label] = [
        dict(label='Count', value=''),
        dict(label='Per capita', value='per 100,000'), 
    #   dict(label='Daily change', value='change daily'),
    #   dict(label='New infections (weekly)', value='new_abs'),
    ]
    return col_options, dimensions

def caveat_markdown_text_covid(): 
    return '''

    # [Insert text about selected region here]


    #### Data sources and notes
    ##### COVID-19 data
    - COVID-19 data corresponds to the present or past state of confirmed infections, 
    and does not account for inconsistent [testing](https://fivethirtyeight.com/features/coronavirus-case-counts-are-meaningless/). 
    It is updated daily, and comes from the [New York Times](https://www.nytimes.com/interactive/2020/us/coronavirus-us-cases.html). 
    - Part one of the CDC gate criteria for re-opening requires 14 consecutive days of declining case counts. 
    That's operationalized as the slope across a three-day moving average of the confirmed case count showing consistent decline.
    - "Active" cases are (conservatively) assumed to resolve in four weeks. 
    According to the [WHO](https://www.who.int/dg/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---24-february-2020), 
    most cases are mild and resolve in two weeks, 
    while about 20% of cases more severe cases resolve in 3-6 weeks.
    - Per capita normalization uses 2018 Census data. 

    '''


def caveat_markdown_text(): 
    return '''
    #### Data sources and notes
    ##### COVID-19 data
    - COVID-19 data corresponds to the present or past state of confirmed infections, 
    and does not account for inconsistent [testing](https://fivethirtyeight.com/features/coronavirus-case-counts-are-meaningless/). 
    It is updated daily, and comes from the [New York Times](https://www.nytimes.com/interactive/2020/us/coronavirus-us-cases.html). See other the statistics tab for more information.

    ##### Employee locations
    - Employee office duty locations are included for non-remote employees. 
    - Exam locations come from the Supervision Exam System (SES), and are updated in real time. 
    - Data from the Travel Office improves the accuracy of upcoming exam travel plans, 
    but may not reflect final travel decisions.
 
    ###### Caveats
    - The SES exam data may over-represent actual planned onsite travel. 
    Examiners may be onsite or offsite, and some are scheduled to multiple exams. 
    Exams are all assumed to be 8 weeks long for each examiner 
    because of inaccuracies in individual examiner reporting in the SES.
    - Some exams involve discretionary monitoring, 
    and offsite travel to those may only be captured in travel data.

    '''
#    - [ ] Data from the Travel Office has not been incorporated, but will improve the accuracy of upcoming exam travel plans.
#    - [ ] External events and conferences have not been incorporated.

