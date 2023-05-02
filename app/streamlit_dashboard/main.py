import re

import plotly.subplots
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import requests

base_uri = 'http://localhost:8080/' # https://mvp-zr5ic3h7sq-as.a.run.app
train_model_api = '/api/v1/lt_reliability/train'
forecast_cycle_api = '/api/v1/lt_reliability/train'
forecast_mileage_api = '/api/v1/lt_reliability/forecast_by_mileage'
data_sourcing_batch_api = '/api/v1/data_sourcing/batch'
data_sourcing_trx_api = '/api/v1/data_sourcing/transaction'
query_all_hist_api = '/api/v1/data_sourcing/history'

st.title('Long Term Reliability Model')
st.text('In REAMS, long term reliability for train system is measured by failure per \nkilometers /'
        '(FPMK). Simulate and forecast long term reliability is of \ngreat importance in renewal /'
        'planning')

uploaded_file = st.file_uploader('upload sample FPMK history')

fpmk_hist = pd.read_csv(uploaded_file)
st.dataframe(fpmk_hist.head())

