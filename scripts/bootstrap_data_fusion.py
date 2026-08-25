#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, html, json, os, re, shutil, subprocess
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
LANDING="https://www.sfmta.com/getting-around/drive-park/demand-responsive-pricing/sfpark-evaluation"


def sql_file(path: Path):
    dsn=os.environ.get("DATABASE_URL"); psql=shutil.which("psql")
    if not dsn or not psql: raise RuntimeError("DATABASE_URL and psql are required")
    subprocess.run([psql,dsn,"-v","ON_ERROR_STOP=1","-f",str(path)],check=True)


def resolve_link(fragment: str) -> str:
    req=Request(LANDING,headers={"User-Agent":"sf-parking-data-fusion/1.0"})
    with urlopen(req,timeout=60) as r: text=r.read().decode("utf-8","replace")
    pattern=re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',re.I|re.S)
    for href,anchor in pattern.findall(text):
        label=re.sub(r"<[^>]+>"," ",html.unescape(anchor)); label=re.sub(r"\s+"," ",label).strip()
        if fragment.lower() in label.lower(): return urljoin(LANDING,href)
    raise RuntimeError(f"Could not resolve SFMTA link: {fragment}")


def download(url:str,dest:Path):
    if dest.exists() and dest.stat().st_size>0: return
    dest.parent.mkdir(parents=True,exist_ok=True)
    req=Request(url,headers={"User-Agent":"sf-parking-data-fusion/1.0"})
    with urlopen(req,timeout=120) as r, dest.open("wb") as f:
        while True:
            b=r.read(1024*1024)
            if not b: break
            f.write(b)


def norm(s): return re.sub(r"[^a-z0-9]+","_",s.strip().lower()).strip("_")


def pipe_csv(path:Path, table:str, columns:list[str], field_map:dict[str,str]):
    dsn=os.environ.get("DATABASE_URL"); psql=shutil.which("psql")
    if not dsn or not psql: raise RuntimeError("DATABASE_URL and psql are required")
    proc=subprocess.Popen([psql,dsn,"-v","ON_ERROR_STOP=1","-c",f"\\copy {table} ({','.join(columns)}) FROM STDIN WITH (FORMAT csv)"],stdin=subprocess.PIPE,text=True,encoding="utf-8")
    assert proc.stdin is not None
    w=csv.writer(proc.stdin,lineterminator="\n")
    with path.open("r",encoding="utf-8-sig",newline="",errors="replace") as f:
        reader=csv.DictReader(f)
        mapped={norm(k):v for k,v in next(iter([reader.fieldnames or []]),[]) if False}
        reader.fieldnames=[norm(x) for x in (reader.fieldnames or [])]
        missing=[src for src in field_map.values() if src and src not in reader.fieldnames]
        if missing: raise RuntimeError(f"{path.name} missing fields: {missing}")
        for row in reader:
            w.writerow([row.get(field_map[c],"") for c in columns])
    proc.stdin.close(); rc=proc.wait()
    if rc: raise RuntimeError(f"Import failed: {path}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--download-historical",action="store_true")
    ap.add_argument("--sensor-csv",type=Path)
    ap.add_argument("--smart-payments-csv",type=Path)
    args=ap.parse_args()

    sql_file(ROOT/"db/migrations/2026-08-25_data_fusion.sql")
    sql_file(ROOT/"db/migrations/2026-08-25_sfpark_historical_sessions.sql")

    if args.download_historical:
        raw=ROOT/"data/raw"; raw.mkdir(parents=True,exist_ok=True)
        args.sensor_csv=raw/"SFpark_ParkingSensorData_HourlyOccupancy_20112013.csv"
        args.smart_payments_csv=raw/"SFpark_MeterData_PaymentTransactions_Smart_20112013.csv"
        download(resolve_link("SFpark Parking Sensor Data Hourly Occupancy 2011 - 2013"),args.sensor_csv)
        download(resolve_link("SFpark Meter Data Payment Transactions Smart"),args.smart_payments_csv)

    if args.sensor_csv:
        cols=["block_id","street_name","block_num","street_block","area_type","pm_district_name","rate","rate_type","start_time_local","total_time","total_occupied_time","total_vacant_time","total_unknown_time","op_time","op_occupied_time","op_vacant_time","op_unknown_time","nonop_time","nonop_occupied_time","nonop_vacant_time","nonop_unknown_time","gmp_time","gmp_occupied_time","gmp_vacant_time","gmp_unknown_time","comm_time","comm_occupied_time","comm_vacant_time","comm_unknown_time"]
        pipe_csv(args.sensor_csv,"sfpark_sensor_hourly",cols,{c:c for c in cols[:-1]}|{"start_time_local":"start_time_dt","comm_unknown_time":"comm_unknown_time"})
    if args.smart_payments_csv:
        cols=["parking_management_district","collected_date_local","street_block","post_id","payment_type","net_amount_paid","session_start_utc","session_end_utc"]
        fields={"parking_management_district":"parking_management_district","collected_date_local":"date","street_block":"street_and_block","post_id":"post_id","payment_type":"payment_type","net_amount_paid":"net_amount_paid","session_start_utc":"session_start_date","session_end_utc":"session_end_date"}
        # normalize headers in a lightweight pre-pass; SFMTA names are documented in its meter guide.
        tmp=args.smart_payments_csv.with_suffix(".normalized.csv")
        with args.smart_payments_csv.open("r",encoding="utf-8-sig",newline="",errors="replace") as f, tmp.open("w",encoding="utf-8",newline="") as o:
            r=csv.DictReader(f); r.fieldnames=[norm(x) for x in (r.fieldnames or [])]; w=csv.writer(o,lineterminator="\n"); w.writerow(cols)
            for row in r: w.writerow([row.get(fields[c],"") for c in cols])
        # Import the normalized file against canonical column names.
        pipe_csv(tmp,"sfpark_payment_session_historical",cols,{c:c for c in cols}); tmp.unlink()

    print("Data-fusion bootstrap complete.")

if __name__=="__main__": main()
