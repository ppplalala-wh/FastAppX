import re

import plotly.subplots
import streamlit as st
import requests
import pandas as pd
import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import altair as alt

base_uri = 'https://mvp-zr5ic3h7sq-as.a.run.app/' # https://mvp-zr5ic3h7sq-as.a.run.app
train_model_api = '/api/v1/lt_reliability/train'
forecast_cycle_api = '/api/v1/lt_reliability/forecast_by_cycles'
forecast_mileage_api = '/api/v1/lt_reliability/forecast_by_mileage'
forecast_api = '/api/v1/lt_reliability/forecast'
data_sourcing_batch_api = '/api/v1/data_sourcing/batch'
data_sourcing_trx_api = '/api/v1/data_sourcing/transaction'
query_all_hist_api = '/api/v1/data_sourcing/history'


st.title('Long Term Reliability Model')
st.text('In REAMS, long term reliability for train system is measured by failure per \nkilometers /'
        '(FPMK). Simulate and forecast long term reliability is of \ngreat importance in renewal /'
        'planning')

def plot_fpmk(fpmk_hist:pd.DataFrame, fpmk_full:pd.DataFrame=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, )
    fig.add_trace(go.Bar(
        x=fpmk_hist['date'],
        y=fpmk_hist['fpmk'], name='fpmk'), row=1, col=1)
    if fpmk_full is not None:
        fig.add_trace(go.Scatter(
            x=fpmk_full['date'],
            y=fpmk_full['fpmk'],
            mode='lines',
            name='predicted_fpmk'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=fpmk_full['date'],
            y=fpmk_full['mileage'],
            mode='lines',
            name='mileage'
        ), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=fpmk_hist['date'],
            y=fpmk_hist['mileage'],
            mode='lines',
            name='mileage'
        ), row=2, col=1)
    # edit axis labels
    fig['layout']['yaxis']['title'] = 'FPMK'
    fig['layout']['yaxis2']['title'] = 'Mileage(km)'

    # Plot!
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Data Sourcing")
sec1, sec2 = st.columns(2)
with sec1:
    st.text('Upload sample FPMK history by batch')
    uploaded_file = st.file_uploader('')

    upload_response = requests.post(base_uri+data_sourcing_batch_api, files={'file': uploaded_file})

    all_data = requests.get(base_uri + query_all_hist_api)
    fpmk_hist = pd.DataFrame(data=json.loads(all_data.content))
    fpmk_hist['date'] = pd.to_datetime(fpmk_hist['date'])

with sec2:
    st.text('Manually Add FPMK records')
    with st.form(key='manual_fpmk'):
        col1, col2 = st.columns(2)
        with col1:
            system_input = st.radio(
                label="System",
                options=["RSC"],
                label_visibility='visible'
            )
            subsystem_input = st.radio(
                label="Sub-System",
                options=["DOR"],
                label_visibility='visible'
            )
        with col2:
            fpmk_input = st.number_input(
                label="FPMK",
                label_visibility='visible',
            )
            mileage_input = st.number_input(
                label="Mileage(km)",
                label_visibility='visible'
            )
            date_input = st.date_input(
                label="Date",
                label_visibility='visible'
            )
        submit_button = st.form_submit_button(label='Upload')
        if submit_button:
            data = {
                'system': system_input,
                'subsystem': subsystem_input,
                'mileage': int(mileage_input),
                'unit': 'km',
                'fpmk': fpmk_input,
                'date': date_input.strftime("%Y-%m-%d")
            }
            response = requests.post(base_uri+data_sourcing_trx_api, params=data)
            if response.status_code == '200':
                st.text('Data upload successfully')

st.subheader('Historical FPMK data')
with st.form(key='historical_fpmk'):
    hist_button= st.form_submit_button(label='Refresh')
    if hist_button or submit_button or upload_response:
        all_data = requests.get(base_uri + query_all_hist_api)
        fpmk_hist = pd.DataFrame(data=json.loads(all_data.content))
        fpmk_hist['date'] = pd.to_datetime(fpmk_hist['date'])
        plot_fpmk(fpmk_hist)

st.subheader('Train model')
with st.form(key='model_training'):
    end_time = st.slider(
        label='Select train data range',
        value=fpmk_hist['date'].max().date(),
        min_value=fpmk_hist['date'].min().date(),
        max_value=fpmk_hist['date'].max().date(),
    )
    end_date = pd.to_datetime(end_time)
    fpmk_train = fpmk_hist[fpmk_hist['date'] <= end_date].copy()
    fpmk_train.loc[:, 'date'] = fpmk_train['date'].dt.strftime('%Y-%m-%d')
    oem_life_input = st.number_input(
        label="OEM life(km)",
        label_visibility='visible'
    )
    strat_train_button = st.form_submit_button(label='Start Training')
    if strat_train_button:
        train_data_list = []
        for item in json.loads(fpmk_train.to_json(orient='records')):
            train_data_list.append(item)
        response = requests.post(base_uri+train_model_api,
                                 json=train_data_list,
                                 params={'design_oem_mileage': int(oem_life_input)})
        print(response.status_code)
        if response.status_code == 200:
            st.text('model training completed.')
            response_content = json.loads(response.content)
            st.text(f"best fitted parameter: {response_content['best_params']}")
            st.text(f"Model Id: {response_content['model_id']}")

st.subheader('Serve model')
with st.form(key='model_serving'):
    mod_id_input = st.text_input(
        label="model_id",
        label_visibility='visible',
    )
    start_fcst_button = st.form_submit_button(label='Start Forecasting')
    if start_fcst_button:
        response = requests.post(base_uri + forecast_cycle_api,
                                 params={"mod_id": mod_id_input, "no_fcst_cycles": 100})
        print(f"forecast response: {response.status_code}")
        fpmk_fcst = pd.DataFrame(data=json.loads(response.content))
        print(fpmk_fcst.head())
        fpmk_fcst['date'] = pd.to_datetime(fpmk_fcst['date'])
        fpmk_fcst.dropna(inplace=True)
        full_data = pd.concat([fpmk_hist, fpmk_fcst]).drop_duplicates(subset=['date'])
        full_data_list = []
        for item in json.loads(full_data.to_json(orient='records')):
            full_data_list.append(item)
        response = requests.post(base_uri+forecast_api,
                                 json=full_data_list,
                                 params={"mod_id": mod_id_input})
        full = pd.DataFrame(data=json.loads(response.content))
        full['date'] = pd.to_datetime(full['date'])
        print(full.head())
        plot_fpmk(fpmk_hist=fpmk_hist, fpmk_full=full)






