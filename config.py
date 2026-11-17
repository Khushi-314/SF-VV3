"""Explicit factory assumptions and live data source."""
SHEET_URL = ('https://docs.google.com/spreadsheets/d/e/'
 '2PACX-1vQFCvr2DfsoEuYRtgV-Y4xQ1khS5tT4UGqbQMWv7op-SgwkNGj6gF4FLRG4d-Vz5J7EZ-6dOqEf2ywm/'
 'pub?gid=0&single=true&output=csv')
PRODUCTS = ('PR1','PR2','PR3','PR4')
STAGES = ('WS1','WS2','WS3','WS4','WS5','WS6','WS7','WS8')
STAGE_RESOURCES = {
 'WS1':('WS1',), 'WS2':('MC1','MC3'), 'WS3':('MC2','MC4'),
 'WS4':('WS4',), 'WS5':('MC5',), 'WS6':('MC6','MC7'),
 'WS7':('MC8','MC9'), 'WS8':('WS8',)
}
MACHINES = ('MC1','MC2','MC3','MC4','MC5','MC6','MC7','MC8','MC9')
RESOURCES = ('WS1','MC1','MC3','MC2','MC4','WS4','MC5','MC6','MC7','MC8','MC9','WS8')
OBS_COLUMNS = ('WS1','WS2','WS3','WS4','WS5-7','WS8')
# Historical approximate shares based on indicative 52 s, 27 s, 5 s stage times.
# These are assumptions; NOT separately identified from the combined sheet.
FMS_SHARES = {'WS5':52/84, 'WS6':27/84, 'WS7':5/84}
# WS8 is the final modeled workstation, measured directly in the source column WS8.
STAGE_OBS = {'WS1':'WS1','WS2':'WS2','WS3':'WS3','WS4':'WS4',
             'WS5':'WS5-7','WS6':'WS5-7','WS7':'WS5-7','WS8':'WS8'}
B_STATES = {'B1':'Idle','B2':'Setup','B3':'Processing','B4':'Off',
            'B5':'Under maintenance','B6':'Waiting for machine',
            'B7':'Error/Alarm','B8':'Not applicable'}
DEFAULT_MTTF_S = 8*3600.0
DEFAULT_BETA = 2.0
DEFAULT_REPAIR_S = 5*60.0
DEFAULT_REPAIR_CV = 0.5
