import re

import plotly.subplots
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

reliability_url = 'http://localhost:8080/'

st.title('Long Term Reliability Model')
st.text('In REAMS, long term reliability for train system is measured by failure per \nkilometers /'
        '(FPMK). Simulate and forecast long term reliability is of \ngreat importance in renewal /'
        'planning')

uploaded_file = st.file_uploader('upload sample FPMK history')

fpmk_hist = pd.read_csv(uploaded_file)
st.dataframe(fpmk_hist.head())

