#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:13:35 2020

@author: cgiordano
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:13:55 2020

@author: cgiordano
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:04:34 2020

@author: cgiordano
"""

def moving_average(x, w):
    import numpy as np
    return np.convolve(x, np.ones(w), 'valid') / w

def data_preparation():
    import pandas as pd
    import numpy as np
    from astropy.time import Time
    from scipy.interpolate import interp1d
#    import matplotlib.pyplot as plt
    import math
    import copy
    import glob
    import subprocess
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    #Names of the MASS and Meteo files
    meteo_dir = '/Users/cgiordano/Documents/Travail/WRF/Calern_ML/Data/CATS/'
    meteo_tag = 'meteo_cats_*.csv'
    gdimm_dir = '/Users/cgiordano/Documents/Travail/WRF/Calern_ML/Data/GDIMM_tmp/'
    gdimm_tag = 'new_r0Alt_2*.csv'
    pml_dir = '/Users/cgiordano/Documents/Travail/WRF/Calern_ML/Data/PBL_tmp/'
    pml_tag   = 'new_Cn2_2*.csv'

    GDIMM_files = sorted(glob.glob(gdimm_dir+'/'+gdimm_tag))
    nGDIMM = len(GDIMM_files)
    PML_files = sorted(glob.glob(pml_dir+'/'+pml_tag))
    nPML = len(PML_files)
    METEO_files = np.array(glob.glob(meteo_dir+'/'+meteo_tag))
    filetest = np.array(['-'.join(item.split('_')[-1].split('.')[0].split('-')[::-1]) for item in METEO_files])
    idx = np.argsort(filetest)
    METEO_files = [item for item in METEO_files[idx]]
    nMETEO = len(METEO_files)
        
    #Read CSV files (MASS and Meteo) by using pandas package
    print('Read CSV files')
 
    print('Read GDIMM files')
    idimm = 0
    for file in GDIMM_files:
        count = len(open(file).readlines(  ))
        if count > 23:
            print('GDIMM: ', idimm,'/',nGDIMM)
            if idimm == 0:
                gdimmtmp = pd.read_csv(file,header=20,parse_dates={'Datetmp':['Date','Time_UT']})
            else:
                gdimmtmp = gdimmtmp.append(pd.read_csv(file,header=20,parse_dates={'Datetmp':['Date','Time_UT']}))
            idimm += 1
    gdimmtmp = gdimmtmp.rename(columns={"Datetmp": "Date", "Tau0(ms)": "Tau0"})
    gdimmtmp.insert(16, 'Seeing', 0.5*(gdimmtmp['epsT']+gdimmtmp['epsL']))
    index_gdimm_to_remove = gdimmtmp.columns.drop(['Seeing', 'Date', 'Isop', 'Tau0'])
    gdimmtmp = gdimmtmp.drop(columns=index_gdimm_to_remove)
  
    ipml = 0
    for file in PML_files:
        count = len(open(file).readlines(  ))
        if count > 23:
            print('PML: ', ipml,'/',nPML)
            if ipml == 0:
                header = pd.read_csv(file, delimiter=';')
                dh_pml = ([s for s in header[header.columns[0]] if "thickness" in s][0]).split(',')
                h_pml = ([s for s in header[header.columns[0]] if "thickness" in s][0]).split(',')
                while('' in dh_pml): dh_pml.remove('')
                while('' in h_pml): h_pml.remove('')
                dh_pml = dh_pml[1:]
                dh_pml = np.array([int(x) for x in dh_pml])
                h_pml = h_pml[1:]
                h_pml = np.array([int(x) for x in h_pml])
                iheader = [i for i in range(len(header[header.columns[0]])) if 'Date' in header[header.columns[0]][i]][0]+1
                pmltmp = pd.read_csv(file,header=iheader,parse_dates={'Datetmp':['Date','Time_UT']})
            else:
                header = pd.read_csv(file, delimiter=';')
                iheader = [i for i in range(len(header[header.columns[0]])) if 'Date' in header[header.columns[0]][i]][0]+1
                pmltmp = pmltmp.append(pd.read_csv(file,header=iheader,parse_dates={'Datetmp':['Date','Time_UT']}))
            ipml += 1
    pmltmp = pmltmp.rename(columns={"Datetmp": "Date", "Tau0(ms)": "Tau0", "Seeing[arcsec]": "Seeing", "Isop[arcsec]": "Isop"})
    # index_pml_to_remove = pmltmp.columns.drop(['Seeing', 'Date', 'Isop', 'Cn2_ground'])
    index_pml_to_remove = pmltmp.columns.drop(['Date', 'Seeing', 'Cn2_ground', 'Cn2_150', 'Cn2_250'])
    pmltmp = pmltmp.drop(columns=index_pml_to_remove)

    print('Read meteo files')
    imeteo = 0
    for file in METEO_files:
        #print(file)
        count = len(open(file).readlines(  ))
        if count > 4:
            print('Meteo: ',imeteo,'/',nMETEO)
            if imeteo == 0:
                meteotmp = pd.read_csv(file,header=1,parse_dates={'Date':['YYYY/MM/JJ',' hh:mn:sec(HL)']})
            else:
                meteotmp = meteotmp.append(pd.read_csv(file,header=1,parse_dates={'Date':['YYYY/MM/JJ',' hh:mn:sec(HL)']}))
            imeteo += 1
        
    meteotmp = meteotmp.rename(str.strip,axis='columns')   
    meteotmp = meteotmp.rename(columns={"outTemp": "Temperature", "outHumidity": "Humidity"})
    index_meteo_to_remove = meteotmp.columns.drop(['Date', 'pressure', 'Temperature', 'Humidity', 'windSpeed', 'windDir'])
    meteotmp = meteotmp.drop(columns=index_meteo_to_remove)
    
    #Put pandas output in dictionaries    
    label_gdimm = gdimmtmp.columns
    label_pml = pmltmp.columns
    label_meteo = meteotmp.columns

    gdimm = {}
    pml = {}
    meteo = {}
    
    for key in label_gdimm:
        gdimm[key] = gdimmtmp[key].to_numpy()
        if gdimm[key].dtype == 'float64':
            gdimm[key][np.where(gdimm[key] < -1000.)] = np.nan
    for key in label_pml:
        pml[key] = pmltmp[key].to_numpy()
        if pml[key].dtype == 'float64':
            print(key, len(pml[key][np.where(pml[key] < -1000.)]))
            pml[key][np.where(pml[key] < -1000.)] = np.nan
    for key in label_meteo:
        meteo[key] = meteotmp[key].to_numpy()
        if meteo[key].dtype == 'float64':
            print(key, len(meteo[key][np.where(meteo[key] < -1000.)]))
            meteo[key][np.where(meteo[key] < -1000.)] = np.nan

    
    #Filtering seeing values higher than 3 arcsec and replace it by an average over closest seeing
    test = np.copy(gdimm['Seeing'])
    idx_seeing = np.flatnonzero(test > 3.)
    test[idx_seeing] = 0.5*(test[idx_seeing-1]+test[idx_seeing+1])
    
    for i in idx_seeing:
        test[i] = (sum(test[i-6:i+7])-test[i])/12
    gdimm['Seeing'] = test

    #Add date and time in an exploitable format in dictionaries
    print('Compute dates')
    gdimm['Date']  = Time(gdimm['Date'].astype('datetime64'))
    pml['Date']  = Time(pml['Date'].astype('datetime64'))
    meteo['Date']  = Time(meteo['Date'].astype('datetime64'))
    
    
    
    #Interpolation of parameters to a given time step
    #We choose a time step of 2minutes which is close to the GDIMM resolution
    print('Resampling of datas to dt=2minutes')
    dt = 5.

    gdimm_resamp  = copy.deepcopy(gdimm)
    date_gdimm_resamp  = np.unique(np.round(gdimm['Date'].jd * 24. * 60. / dt)*dt/24./60.)

    pml_resamp  = copy.deepcopy(pml)
    date_pml_resamp  = np.unique(np.round(pml['Date'].jd * 24. * 60. / dt)*dt/24./60.)

    meteo_resamp = copy.deepcopy(meteo)
    date_meteo_resamp = np.unique(np.round(meteo['Date'].jd * 24. * 60. / dt)*dt/24./60.)

    #Selection of case where there is a simultaneity between GDIMM, PML and METEO data
    # idx_gdimm_pml = np.flatnonzero(np.isin(date_gdimm_resamp, date_pml_resamp))
    # date_gdimm_resamp = date_gdimm_resamp[idx_gdimm_pml]    

    idx_gdimm_meteo = np.flatnonzero(np.isin(date_gdimm_resamp, date_meteo_resamp))
    date_gdimm_resamp = date_gdimm_resamp[idx_gdimm_meteo]    
 
    idx_meteo_gdimm = np.flatnonzero(np.isin(date_meteo_resamp, date_gdimm_resamp))
    date_meteo_resamp = date_meteo_resamp[idx_meteo_gdimm]

    # idx_pml_meteo = np.flatnonzero(np.isin(date_pml_resamp, date_meteo_resamp))
    # date_pml_resamp = date_pml_resamp[idx_pml_meteo]    
 
    # idx_meteo_pml = np.flatnonzero(np.isin(date_meteo_resamp, date_pml_resamp))
    # date_meteo_resamp = date_meteo_resamp[idx_meteo_pml]

    date_resamp = date_meteo_resamp

    #Set startdate and enddate
    date_init = math.floor(min(date_meteo_resamp*24.))/24.
    date_end  = math.ceil(max(date_meteo_resamp*24.))/24.
    
    #Create temporal vector with dt sampling
    date_set = np.arange(date_init*24*60., date_end*24.*60.+dt, dt) / 24. / 60.
    date_set = Time(date_set, format='jd')
    
    #extract year, cos(day), sin(day), hour
    data = {}
    data['Date'] = date_set
    data['year'] = date_set.ymdhms.year
    data['day'] = np.array([int(date_set.yday[i].split(':')[1])/366. for i in np.arange(len(date_set.yday))])
    data['cday'] = np.cos(data['day']*2*Pi)
    data['sday'] = np.sin(data['day']*2*Pi)
    data['hour'] = (date_set.ymdhms.hour+date_set.ymdhms.minute/60+date_set.ymdhms.second/60./60.)/24.
    data['chour'] = np.cos(data['hour']*2*Pi)
    data['shour'] = np.sin(data['hour']*2*Pi)

    #index where date corresponds to a real measurements
    idx_resamp = np.flatnonzero(np.isin(date_set.jd, date_resamp))
     
    #interpolate measurements to date_set and put NaN where there is no measurements
    for key, v in gdimm.items():
        if 'Date'.lower() not in key.lower():
            print(key)
            finterp = interp1d(gdimm['Date'].jd, v, fill_value='extrapolate')
            data[key] = np.full(len(date_set.jd),np.nan)
            data[key][idx_resamp] = finterp(date_resamp) 

    # for key, v in pml.items():
    #     if 'Date'.lower() not in key.lower():
    #         print(key)
    #         finterp = interp1d(pml['Date'].jd, v, fill_value='extrapolate')
    #         data[key] = np.full(len(date_set.jd),np.nan)
    #         data[key][idx_resamp] = finterp(date_resamp) 
            
    for key, v in meteo_resamp.items():
        if 'Date'.lower() not in key.lower():
            print(key)
            finterp = interp1d(meteo['Date'].jd, v, fill_value='extrapolate')
            data[key] = np.full(len(date_set.jd),np.nan)
            data[key][idx_resamp] = finterp(date_resamp)                     
            
    #Creation of matrix data covering the entire set 

    #Separation of training sets and test sets
    #Training set: X=2h of measurements Y=2h in the future of measurements
             
    i = 0
    nval2h = round(120/dt)
    xtraining = []
    ytraining = []
    ytrainingisop = []
    ytrainingtau0 = []
    xtest = []
    ytest = []
    ytestisop = []
    ytesttau0 = []
    keys = data.keys()
    xkey = []
    ykey = []
    ykeyisop = []
    ykeytau0 = []
    list_key_to_avoid = ['Date', 'year', 'day', 'hour','cday', 'sday', 'chour', 'shour']
    list_key_to_avoid = [item.lower() for item in list_key_to_avoid]
    while i <= len(data['Date'])-2*nval2h:
        print(i,'/',len(data['Date']))
        xtmp = np.array([])
        ytmp = np.array([])
        xtmp2 = np.array([])
        ytmp2 = np.array([])
        ytmpisop = np.array([])
        ytmp2isop = np.array([])
        ytmptau0 = np.array([])
        ytmp2tau0 = np.array([])
        for key in keys:
            if key.lower() in ['cday', 'sday', 'chour', 'shour']:
                if i == 0:
                    xkey.append(key)
                xtmp = np.append(xtmp,data[key][i])
                if i+3*nval2h <= len(data['Date']):
                    xtmp2 = np.append(xtmp2,data[key][i+nval2h])
            if key.lower() not in list_key_to_avoid:
                if i == 0:
                    xkey.append(key)
                xtmp = np.append(xtmp, data[key][i:i+nval2h])
                if i+3*nval2h <= len(data['Date']):
                    xtmp2 = np.append(xtmp2, data[key][i+nval2h:i+2*nval2h])
            if 'seeing' in key.lower():
                if i == 0:
                    ykey.append(key)
                ytmp = np.append(ytmp, data[key][i+nval2h:i+2*nval2h])
                if i+3*nval2h <= len(data['Date']): 
                    ytmp2 = np.append(ytmp2, data[key][i+2*nval2h:i+3*nval2h])
            if 'isop' in key.lower():
                if i == 0:
                    ykeyisop.append(key)
                ytmpisop = np.append(ytmpisop, data[key][i+nval2h:i+2*nval2h])
                if i+3*nval2h <= len(data['Date']): 
                    ytmp2isop = np.append(ytmp2isop, data[key][i+2*nval2h:i+3*nval2h])
            if 'tau0' in key.lower():
                if i == 0:
                    ykeytau0.append(key)
                ytmptau0 = np.append(ytmptau0, data[key][i+nval2h:i+2*nval2h])
                if i+3*nval2h <= len(data['Date']): 
                    ytmp2tau0 = np.append(ytmp2tau0, data[key][i+2*nval2h:i+3*nval2h])
                 
        xtraining.append(xtmp)
        ytraining.append(ytmp)
        ytrainingisop.append(ytmpisop)
        ytrainingtau0.append(ytmptau0)

        if len(xtmp2) > 0:
            xtest.append(xtmp2)
        if len(ytmp2) > 0:
            ytest.append(ytmp2)
        if len(ytmp2isop) > 0:
            ytestisop.append(ytmp2isop)
        if len(ytmp2tau0) > 0:
            ytesttau0.append(ytmp2tau0)

        i = i+2*nval2h
    
    xtraining = np.array(xtraining)
    ytraining = np.array(ytraining)
    ytrainingisop = np.array(ytrainingisop)
    ytrainingtau0 = np.array(ytrainingtau0)
    
    xtest = np.array(xtest)
    ytest = np.array(ytest)
    ytestisop = np.array(ytestisop)
    ytesttau0 = np.array(ytesttau0)
    
    bins = np.arange(date_set.isot[0],date_set.isot[-1], dtype='datetime64[M]')
    bins = np.append(bins, [bins[-1]+1,bins[-1]+2])
    idx = np.isnan(data['Seeing'])
    fig, hist = plt.subplots(1,1)
    mpl_data = mdates.date2num(date_set.datetime[~idx])
    mpl_bins = mdates.date2num(bins)
    hist.hist(mpl_data, bins=mpl_bins, edgecolor = 'black')
    hist.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]))
    hist.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.xticks(rotation=45)
    plt.ylabel('Number of measurements')
    plt.show()
    
    return xtraining, ytraining, xtest, ytest, xkey, ykey, ytrainingisop, ytestisop, ytrainingtau0, ytesttau0