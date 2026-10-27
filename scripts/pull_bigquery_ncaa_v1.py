"""Extract tables from bigquery-public-data.ncaa_basketball to local parquet.

Private internal research use only (Sportradar notice; user ruling 2026-10-27).
Installed versions (pin on install):
  pip install google-cloud-bigquery==3.46.2 google-cloud-bigquery-storage==2.43.0 \
      pydata-google-auth==1.9.1 db-dtypes==1.9.0

Usage (from repo root, .venv/Scripts/python.exe):
  scripts/pull_bigquery_ncaa_v1.py --project P --auth
  scripts/pull_bigquery_ncaa_v1.py --project P --list
  scripts/pull_bigquery_ncaa_v1.py --project P --schema mbb_pbp_sr
  scripts/pull_bigquery_ncaa_v1.py --project P --pull mbb_pbp_sr --dry-run
  scripts/pull_bigquery_ncaa_v1.py --project P --pull mbb_pbp_sr [--max-gb 50]

Output: data/raw/bigquery_ncaa/<table>/<table>_<season>.parquet (or
<table>_chunkNN.parquet when the table has no season column; chunks hash game_id).
Columns are kept as-is (event_coord_x / event_coord_y untransformed).
Manifest: data/raw/bigquery_ncaa/manifest.json (hoopR manifest format + query,
bytes_processed). Sandbox allowance: 1 TiB processed per month; bytes are printed.
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
import zlib
from pathlib import Path

DATASET = "bigquery-public-data.ncaa_basketball"
SCOPES = ["https://www.googleapis.com/auth/bigquery.readonly"]
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "bigquery_ncaa"
N_CHUNKS = 20
SEASON_CANDIDATES = ("season", "game_season", "season_id")


_REAL_OPEN = None


def _open_url(url, *a, **k):
    """Print the full auth URL on its own line first, then open it with webbrowser.open."""
    print("Open this URL in a browser on THIS machine if the page does not open or errors:")
    print(url, flush=True)
    try:
        return _REAL_OPEN(url)
    except Exception:
        return False


SA_KEY_DEFAULT = Path.home() / ".config" / "pydata" / "bigquery_sa_key.json"
SA_KEY = None      # set from --sa-key in main(); falls back to SA_KEY_DEFAULT when that file exists
AUTH_PORT = 8085   # fixed port so a Remote-SSH user can forward it (VS Code Ports panel)


def _sa_key_path():
    if SA_KEY:
        return Path(SA_KEY)
    return SA_KEY_DEFAULT if SA_KEY_DEFAULT.exists() else None


def get_credentials(interactive=False):
    """Three ways in, tried in this order:
    1. a service-account JSON key (--sa-key, or the default path above): no browser at all,
       the right choice on a headless box;
    2. cached user credentials from an earlier --auth;
    3. interactive --auth: prints the Google URL, listens on 127.0.0.1:AUTH_PORT for the
       redirect, never opens a browser itself. On a headless box forward that port
       (VS Code: Ports panel -> Forward a Port -> 8085) and open the printed URL on the laptop.
    """
    import pydata_google_auth
    import pydata_google_auth.cache as pcache
    key = _sa_key_path()
    if key is not None:
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_file(str(key), scopes=SCOPES)
    if interactive:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from pydata_google_auth.auth import DESKTOP_CLIENT_ID, DESKTOP_CLIENT_SECRET
        cfg = {"installed": {"client_id": DESKTOP_CLIENT_ID, "client_secret": DESKTOP_CLIENT_SECRET,
                             "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                             "token_uri": "https://oauth2.googleapis.com/token",
                             "redirect_uris": [f"http://localhost:{AUTH_PORT}/"]}}
        flow = InstalledAppFlow.from_client_config(cfg, scopes=SCOPES)
        print(f"Listening on http://localhost:{AUTH_PORT}/ for the redirect. "
              f"Open the URL below in a browser that can reach this machine's localhost:{AUTH_PORT} "
              f"(forward the port first if you are on Remote SSH).", flush=True)
        creds = flow.run_local_server(host="localhost", bind_addr="127.0.0.1", port=AUTH_PORT,
                                      open_browser=False,
                                      authorization_prompt_message="AUTH URL: {url}",
                                      success_message="Signed in. You can close this tab.")
        pcache.ReadWriteCredentialsCache().save(creds)
        return creds
    # cached only: never opens a browser
    if pcache.ReadWriteCredentialsCache().load() is None:
        sys.exit("No cached Google credentials and no service-account key. Run once: "
                 "scripts/pull_bigquery_ncaa_v1.py --project P --auth   (or pass --sa-key KEY.json)")
    return pydata_google_auth.get_user_credentials(SCOPES)


def get_client(project, interactive=False):
    from google.cloud import bigquery
    return bigquery.Client(project=project, credentials=get_credentials(interactive))


def pick_season_column(col_names):
    lower = {c.lower(): c for c in col_names}
    for cand in SEASON_CANDIDATES:
        if cand in lower:
            return lower[cand]
    return None


def partition_frame(df, season_col=None, n_chunks=N_CHUNKS):
    """Yield (label, subframe): by season column, else crc32(game_id) % n_chunks."""
    if season_col is not None:
        for s, sub in df.groupby(season_col, sort=True, dropna=False):
            yield str(s), sub
    else:
        h = df["game_id"].astype(str).map(lambda x: zlib.crc32(x.encode()) % n_chunks)
        for k, sub in df.groupby(h, sort=True):
            yield f"chunk{int(k):02d}", sub


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def append_manifest(manifest_path, base_dir, file_path, dataset, season, query, bytes_processed):
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    rel = Path(file_path).relative_to(base_dir).as_posix()
    m[rel] = {
        "bytes": Path(file_path).stat().st_size,
        "dataset": dataset,
        "downloaded_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "path": rel,
        "season": season,
        "sha256": sha256_file(file_path),
        "status": "downloaded",
        "query": query,
        "bytes_processed": bytes_processed,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(m, indent=2, sort_keys=True))
    return m[rel]


def cmd_auth(args):
    from google.cloud import bigquery
    creds = get_credentials(interactive=True)
    client = bigquery.Client(project=args.project, credentials=creds)
    acct = None
    try:
        import google.auth.transport.requests as tr
        import requests
        creds.refresh(tr.Request())
        r = requests.get("https://oauth2.googleapis.com/tokeninfo",
                         params={"access_token": creds.token}, timeout=15)
        acct = r.json().get("email")
    except Exception:
        pass
    list(client.query("SELECT 1").result())
    print(f"Signed in. Account: {acct}; project {args.project} usable.")


def cmd_list(args):
    client = get_client(args.project)
    rows = client.query(
        f"SELECT table_id, row_count, size_bytes, last_modified_time "
        f"FROM `{DATASET}.__TABLES__` ORDER BY table_id").result()
    print(f"{'table':40s} {'rows':>14s} {'GB':>9s}  last_modified")
    for r in rows:
        lm = dt.datetime.fromtimestamp(r.last_modified_time / 1000, dt.timezone.utc)
        print(f"{r.table_id:40s} {r.row_count or 0:14,d} {(r.size_bytes or 0)/1e9:9.3f}  {lm:%Y-%m-%d %H:%M}Z")


def table_columns(client, table):
    return client.get_table(f"{DATASET}.{table}").schema


def cmd_schema(args):
    for f in table_columns(get_client(args.project), args.schema):
        print(f"{f.name:40s} {f.field_type:12s} {f.mode}")


def dry_bytes(client, sql):
    from google.cloud import bigquery
    j = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    return j.total_bytes_processed


def cmd_pull(args):
    client = get_client(args.project)
    table = args.pull
    fq = f"`{DATASET}.{table}`"
    cols = [f.name for f in table_columns(client, table)]
    season_col = pick_season_column(cols)
    if season_col is None and "game_id" not in cols:
        sys.exit(f"{table}: no season column and no game_id; cannot partition")
    if season_col:
        seasons = [r[0] for r in client.query(
            f"SELECT DISTINCT {season_col} FROM {fq} ORDER BY 1").result()]
        parts = []
        for s in seasons:
            lit = repr(s) if isinstance(s, str) else str(s)
            parts.append((str(s), f"SELECT * FROM {fq} WHERE {season_col} = {lit}"))
        print(f"season column '{season_col}': {len(parts)} partitions")
    else:
        parts = [(f"chunk{k:02d}",
                  f"SELECT * FROM {fq} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(game_id AS STRING))), {N_CHUNKS}) = {k}")
                 for k in range(N_CHUNKS)]
        print(f"no season column; {N_CHUNKS} game_id hash chunks")
    ests = [(lab, sql, dry_bytes(client, sql)) for lab, sql in parts]
    total = sum(e[2] for e in ests)
    print(f"Estimated bytes processed: {total:,} ({total/1e9:.3f} GB; {total/2**40*100:.4f}% of 1 TiB monthly)")
    if args.dry_run:
        for lab, _, b in ests:
            print(f"  {lab}: {b:,}")
        return
    if total / 1e9 > args.max_gb:
        sys.exit(f"REFUSED: estimate {total/1e9:.2f} GB exceeds --max-gb {args.max_gb}")
    outdir = OUT / table
    outdir.mkdir(parents=True, exist_ok=True)
    used = 0
    for lab, sql, _ in ests:
        job = client.query(sql)
        try:
            df = job.to_dataframe(create_bqstorage_client=True)
        except Exception as e:  # Storage API unavailable -> REST fallback
            print(f"  storage API read failed ({type(e).__name__}); falling back to REST")
            df = job.to_dataframe(create_bqstorage_client=False)
        path = outdir / f"{table}_{lab}.parquet"
        df.to_parquet(path, index=False)
        bp = job.total_bytes_processed or 0
        used += bp
        append_manifest(OUT / "manifest.json", OUT, path, table, lab, sql, bp)
        print(f"  {path.name}: {len(df):,} rows, {path.stat().st_size:,} B, processed {bp:,} B")
    print(f"Total bytes processed this pull: {used:,} ({used/2**40*100:.4f}% of 1 TiB monthly)")


def main():
    global SA_KEY, AUTH_PORT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="GCP sandbox project id")
    ap.add_argument("--auth", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--schema", metavar="TABLE")
    ap.add_argument("--pull", metavar="TABLE")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-gb", type=float, default=50.0)
    ap.add_argument("--port", type=int, default=AUTH_PORT, help="local redirect port for --auth")
    ap.add_argument("--sa-key", metavar="JSON", help="service-account key file (no browser needed)")
    args = ap.parse_args()
    SA_KEY = args.sa_key
    AUTH_PORT = args.port
    if args.auth:
        cmd_auth(args)
    elif args.list:
        cmd_list(args)
    elif args.schema:
        cmd_schema(args)
    elif args.pull:
        cmd_pull(args)
    else:
        ap.error("one of --auth --list --schema --pull required")


if __name__ == "__main__":
    main()
