import lightkurve as lk
import numpy as np
import pandas as pd
import io
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
cap=200
rows=[]
def download_confirmed_planets():
    df=pd.read_csv('Data/cumulative_koi_data.csv')
    confirmed=df[df["koi_disposition"]=="CONFIRMED"][["kepid"]]
    confirmed=confirmed.sample(400)

    for value in tqdm(confirmed["kepid"], desc="Downloading Planets"):



            lc= lk.search_lightcurve(f'KIC {value}', mission="Kepler", cadence="long")
            try:
                    if len(lc)>0:
                        lcs=lc.download_all()
                        lcs=lcs.stitch().remove_nans()
                        lcdf=lcs.to_pandas().reset_index()[["time","flux","flux_err"]]
                        lcdf.to_csv(f'Data/raw/KIC{value}.csv', index=False)
                        rows.append({"kepid": value, "source": 'transit'})
                    else:
                        tqdm.write(f"No data found for KIC {value}")
            except:
               tqdm.write(f"No data found for {value}")


def download_ebs():
    df=pd.read_csv('Data/eb_data.csv')
    ebs_data=df["kepid"].sample(400)
    for value in tqdm(ebs_data, desc="Downloading EBs"):

            lc= lk.search_lightcurve(f'KIC {value}', mission="Kepler", cadence="long")
            try:
                if len(lc)>0:
                    lcs=lc.download_all()
                    lcs=lcs.stitch().remove_nans()
                    lcdf=lcs.to_pandas().reset_index()[["time","flux","flux_err"]]
                    lcdf.to_csv(f'Data/raw/KIC{value}.csv', index=False)
                    rows.append({"kepid": value, "source": 'ebs'})
                else:
                    tqdm.write(f"No data found for {value}")
            except:
                tqdm.write(f"No data found for {value}")


def download_stellar():
    df=pd.read_csv('Data/stellar_data.csv')
    df2=pd.read_csv('Data/cumulative_koi_data.csv')
    df3=pd.read_csv('Data/eb_data.csv')
    quiet_stellar=df[~df["kepid"].isin(df2["kepid"])]
    quiet_stellar=quiet_stellar[~quiet_stellar["kepid"].isin(df3["kepid"])]
    quiet_stellar=quiet_stellar.sample(200)
    for value in tqdm(quiet_stellar["kepid"], desc="Downloading Quiet Stars"):
            lc=lk.search_lightcurve(f"KIC {value}", mission="Kepler", cadence="long")
            try:
                if len(lc)>0:
                    lcs=lc.download_all()
                    lcs=lcs.stitch().remove_nans()
                    lcdf=lcs.to_pandas().reset_index()[["time","flux","flux_err"]]
                    lcdf.to_csv(f'Data/raw/KIC{value}.csv', index=False)
                    rows.append({"kepid": value, "source": 'quiet'})
            except:
                tqdm(f"No data found for {value}")


if __name__=="__main__":
    download_confirmed_planets()
    download_ebs()
    download_stellar()
    print("Successfully downloaded!")
    df = pd.DataFrame(rows)
    df.sort_values(by="source")
    df.to_csv("Data/raw/_target.csv")
    print(f"Manifest created successfully! Found {len(df)} light curves.")
    print(df["source"].value_counts())

