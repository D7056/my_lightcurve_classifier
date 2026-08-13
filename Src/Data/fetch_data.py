import pandas as pd
import requests
import io
import certifi
import time

def fetch_data():
    KOI_URL=("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    "select+kepid,kepoi_name,koi_disposition,koi_period,koi_duration,koi_depth,koi_time0bk+"
    "from+cumulative&format=csv")


    print("Requesting data from NASA TAP API...")

    response= requests.get(KOI_URL, timeout=30)

    try:
        response.raise_for_status()

    except:
        print("The loading failed")

    df=pd.read_csv(io.StringIO(response.text))
    print(f"Successfully downloaded {len(df)} rows!")

    return df

def fetch_eb():
    EB_URL = "https://archive.stsci.edu/kepler/eclipsing_binaries.html"

    print(f"Fetching Kepler Eclipsing Binary Catalog from MAST...")
    last_err=None
    for attempt in range(1,4):
        try:

            response=requests.get(EB_URL, timeout=180, verify=certifi.where())
            response.raise_for_status()

            tables=pd.read_html(io.StringIO(response.text))

            break
        except Exception as e:
            last_err=e

            if attempt<3:
                wait= 5*attempt
                print(f"  ! attempt {attempt}/3 failed ({e}); retrying in {wait}s...")
                time.sleep(wait)
    else:
        print(f"  ! EB fetch failed after retries ({last_err}).")
        print("  (Not required to proceed -- preprocess.py falls back to "
              "synthetic eclipsing_binary injection if this file is missing.)")
        return pd.DataFrame()

    candidates=[t for t in tables if "Kepler ID" in t.columns and "Period" in t.columns and len(t)>1000]
    if not candidates:
        print(f"  ! Found {len(tables)} table(s) on the page but none matched the "
              f"expected EB catalog shape -- the page structure may have changed.")
        return pd.DataFrame()

    df=candidates[0].rename(columns={"Kepler ID":"kepid","Period":"period"})
    df["kepid"]=pd.to_numeric(df["kepid"], errors="coerce")
    df["period"]=pd.to_numeric(df["period"], errors="coerce")
    df=df.dropna(subset=["kepid","period"])
    df["kepid"] = df["kepid"].astype(int)
    final_df=df[["kepid","period"]].drop_duplicates()



    print(f"Successfully downloaded Kepler Eclipsing Binary Catalog of rows {len(final_df)}.")
    return final_df

def fetch_stellar():
    STELLAR_URL = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        "select+kepid+from+keplerstellar&format=csv"
    )
    print("Requesting data from STELLAR API...")
    response= requests.get(STELLAR_URL, timeout=30)
    try:
        response.raise_for_status()
    except:
        print("The loading failed")
    df=pd.read_csv(io.StringIO(response.text))
    df=df.dropna()
    df=df.drop_duplicates()
    print(f"Successfully downloaded {len(df)} rows!")
    return df



stellar_data=fetch_stellar()
stellar_data.to_csv("Data/stellar_data.csv", index=False)