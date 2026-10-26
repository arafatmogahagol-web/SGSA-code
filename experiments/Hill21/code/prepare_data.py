from pathlib import Path
import zipfile,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def run(source):
 blocks=[]
 selected=['wtc_AcWindSp_mean','wtc_GenRpm_mean','wtc_MainSRpm_mean','wtc_PitcPosA_mean','wtc_NacelPos_mean']
 with zipfile.ZipFile(source) as z:
  for month in [1,2,3]:
   base=pd.read_csv(z.open(f'tblSCTurbine_2020_{month:02d}.csv'),usecols=['TimeStamp','StationId']+selected)
   power=pd.read_csv(z.open(f'tblSCTurGrid_2020_{month:02d}.csv'),usecols=['TimeStamp','StationId','wtc_ActPower_mean'])
   if base.duplicated(['TimeStamp','StationId']).any() or power.duplicated(['TimeStamp','StationId']).any():raise ValueError('Duplicate time/node records')
   blocks.append(base.merge(power,on=['TimeStamp','StationId'],how='outer',validate='one_to_one'))
 d=pd.concat(blocks,ignore_index=True);d['TimeStamp']=pd.to_datetime(d.TimeStamp);ids=np.sort(d.StationId.unique())
 if len(ids)!=21:raise ValueError(f'Expected21 turbines, got{len(ids)}')
 # Source timestamps already denote interval ends. The experimental protocol
 # adds a conservative 10-minute availability lag; preserve it for cache parity.
 d['available']=d.TimeStamp+pd.Timedelta(minutes=10);d['bin']=d.available.dt.ceil('30min')
 d['yaw_sin']=np.sin(np.deg2rad(d['wtc_NacelPos_mean']));d['yaw_cos']=np.cos(np.deg2rad(d['wtc_NacelPos_mean']))
 features=selected[:4]+['yaw_sin','yaw_cos','wtc_ActPower_mean']
 grid=pd.date_range(d.bin.min(),d.bin.max(),freq='30min')
 raw=np.stack([d.groupby(['bin','StationId'])[c].mean().unstack('StationId').reindex(index=grid,columns=ids).to_numpy() for c in features],-1).astype('float32')
 T=len(raw);b1=int(.7*T);b2=int(.85*T);coverage=np.isfinite(raw[:b1,:,-1]).mean(0)
 if coverage.min()<.7:raise ValueError(f'Insufficient training label coverage: {coverage.tolist()}')
 mean=np.nanmean(raw[:b1],axis=(0,1));std=np.maximum(np.nanstd(raw[:b1],axis=(0,1)),1e-5);median=np.nanmedian(raw[:b1],axis=(0,1))
 if not np.isfinite(mean).all():raise ValueError('Invalid training statistics')
 x=raw.copy()
 for n in range(len(ids)):x[:,n,:]=pd.DataFrame(x[:,n,:]).ffill().to_numpy()
 x=((np.where(np.isfinite(x),x,median)-mean)/std).astype('float32');y=(raw[:,:,-1]-mean[-1])/std[-1];mask=np.isfinite(y);y=np.nan_to_num(y).astype('float32')
 starts={};counts={}
 for name,a,b,step in [('train',96,b1,4),('validation',b1,b2,24),('test',b2,T,24)]:
  v=np.array([t for t in range(a,b-48+1,step) if mask[t:t+48].any()],dtype='int64');starts[name+'_starts']=v;counts[name]=len(v)
  if len(v)<20:raise ValueError('Too few evaluation windows')
 (ROOT/'data').mkdir(exist_ok=True)
 np.savez_compressed(ROOT/'data/panel.npz',x=x,y=y,mask=mask,raw=raw,mean=mean,std=std,median=median,timestamps=grid.asi8,turbine_ids=ids,**starts)
 audit=dict(dataset='Hill of Towie January-March 2020',source_doi='10.5281/zenodo.14870023',nodes=len(ids),turbine_ids=ids.tolist(),features=features,power_target='wtc_ActPower_mean measured active power in kW; not PowerRef',cadence_minutes=30,history_hours=48,forecast_hours=24,boundaries=[0,b1,b2,T],boundary_dates=[str(grid[b1]),str(grid[b2])],windows=counts,training_power_coverage=coverage.tolist(),target_mask='finite measured labels only; never forward-fill targets into loss',inputs='causal forward fill then training median',normalization='training-only mean/std over time and turbines',availability='source end timestamp plus protocol-specific10min lag, then right-closed30min means',preprocessing_only=True)
 (ROOT/'data_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':
 import sys;run(Path(sys.argv[1]))
