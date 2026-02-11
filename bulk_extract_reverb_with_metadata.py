#!/usr/bin/env python3
"""
Bulk extract the 'Reverberation Measurements' table (<table class="body">)
and append repeated object-level metadata columns to the RIGHT (preserving table structure).

- Fetches HTML via curl -k (works around SSL verification issues)
- Keeps reverb table values as strings (no cleaning of '...' or uncertainties)
- Fixes header row with header=1
- Writes CSV in UTF-8 with BOM (utf-8-sig) so Excel shows τ/σ/β correctly
- Default separator is ';' (usually best for German Excel)

Example:
  python3 bulk_extract_reverb_with_metadata.py --start 1 --end 95 --outdir agnmass_dump/reverb_tables
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
from bs4 import BeautifulSoup


def fetch_html_curl(varname: int, timeout: int = 45) -> str:
    url = f"https://www.astro.gsu.edu/AGNmass/details.php?varname={varname}"
    cmd = ["curl", "-k", "-L", "--max-time", str(timeout), url]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(f"curl failed (code {p.returncode}): {p.stderr.strip()}")
    return p.stdout


def html_to_text_preserve_supsub(html_fragment: str) -> str:
    """
    Convert a small HTML fragment to text while preserving <sup> and <sub> in a readable way.
    Example: 10<sup>7</sup> -> 10^7 ; M<sub>sun</sub> -> M_sun
    """
    frag = BeautifulSoup(html_fragment, "html.parser")

    for sup in frag.find_all("sup"):
        sup.replace_with("^" + sup.get_text(strip=True))
    for sub in frag.find_all("sub"):
        sub.replace_with("_" + sub.get_text(strip=True))

    text = frag.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def br_split_cell_preserve(td) -> List[str]:
    """
    Split a <td> into lines on <br>, preserving sup/sub.
    Returns cleaned text lines.
    """
    inner = td.decode_contents()
    parts = re.split(r"<br\s*/?>", inner, flags=re.IGNORECASE)
    out = []
    for part in parts:
        s = html_to_text_preserve_supsub(part)
        if s:
            out.append(s)
    return out


def parse_metadata(html: str, varname: int) -> Dict[str, Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")
    txt = soup.get_text(" ", strip=True)

    # Object name (big title)
    obj_name = None
    font = soup.find("font", attrs={"size": "6"})
    if font:
        b = font.find("b")
        if b:
            obj_name = b.get_text(strip=True)

    # Alternate Names
    alt_names = None
    for b in soup.find_all("b"):
        if b.get_text(strip=True).startswith("Alternate Names"):
            td = b.parent
            tr = td.parent
            tds = tr.find_all("td")
            try:
                idx = tds.index(td)
                if idx + 1 < len(tds):
                    alt_names = tds[idx + 1].get_text(" ", strip=True) or None
            except ValueError:
                pass
            break

    # RA / Dec / z
    ra = dec = z = None
    m = re.search(r"RA\s*=\s*([0-9:\.]+)\s*Dec\s*=\s*([+\-0-9:]+)\s*z\s*=\s*([0-9.]+)", txt)
    if m:
        ra, dec, z = m.group(1), m.group(2), m.group(3)

    # Distances (nice-to-have)
    dl_mpc = da_mpc = None
    m = re.search(r"D\s*L\s*=\s*([0-9.]+)\s*Mpc.*?D\s*A\s*=\s*([0-9.]+)\s*Mpc", txt)
    if m:
        dl_mpc, da_mpc = m.group(1), m.group(2)

    # f used (should be 4.3 by default)
    f_used = None
    m = re.search(r"calculated using\s*<\s*f\s*>\s*=\s*([0-9.]+)", txt, re.IGNORECASE)
    if m:
        f_used = m.group(1)

    # MBH (Hβ only) and MBH (all lines)
    mbh_hbeta_only = None
    mbh_all_lines = None
    for tr in soup.find_all("tr"):
        t = tr.get_text(" ", strip=True)
        if t.startswith("M BH (Hβ only):"):
            tds = tr.find_all("td")
            if len(tds) >= 2:
                parts = br_split_cell_preserve(tds[1])
                if len(parts) >= 1:
                    mbh_hbeta_only = parts[0]
                if len(parts) >= 2:
                    mbh_all_lines = parts[1]
            break

    # MBH (RM modeling)
    mbh_rm_modeling = None
    mbh_rm_ref = None
    for tr in soup.find_all("tr"):
        t = tr.get_text(" ", strip=True)
        if t.startswith("M BH (RM modeling):"):
            tds = tr.find_all("td")
            if len(tds) >= 2:
                parts = br_split_cell_preserve(tds[1])
                if len(parts) >= 1:
                    mbh_rm_modeling = parts[0]
                if len(parts) >= 2:
                    mbh_rm_ref = parts[1]
            break

    return {
        "varname": str(varname),
        "object_name": obj_name,
        "alternate_names": alt_names,
        "ra": ra,
        "dec": dec,
        "z": z,
        "dl_mpc": dl_mpc,
        "da_mpc": da_mpc,
        "f_used": f_used,
        "mbh_hbeta_only": mbh_hbeta_only,
        "mbh_all_lines": mbh_all_lines,
        "mbh_rm_modeling": mbh_rm_modeling,
        "mbh_rm_ref": mbh_rm_ref,
        "source_url": f"https://www.astro.gsu.edu/AGNmass/details.php?varname={varname}",
    }


def extract_reverb_table_df(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", {"class": "body"})
    if tbl is None:
        raise RuntimeError("Could not find <table class='body'> (Reverberation Measurements table).")

    # Make cells single-line
    for br in tbl.find_all("br"):
        br.replace_with(" ")

    # Use the second header row as column names
    df = pd.read_html(str(tbl), header=1)[0]
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=95)
    ap.add_argument("--outdir", type=str, default="agnmass_dump/reverb_tables")
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--save-html", action="store_true")
    ap.add_argument("--encoding", type=str, default="utf-8-sig", help="utf-8-sig recommended for Excel")
    ap.add_argument("--sep", type=str, default=";", help="';' recommended for German Excel")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    error_log = outdir / "errors.log"
    ok_count = 0
    fail_count = 0

    # Fixed metadata column order (appended to the RIGHT)
    meta_order = [
        "varname", "object_name", "alternate_names", "ra", "dec", "z",
        "dl_mpc", "da_mpc", "f_used",
        "mbh_hbeta_only", "mbh_all_lines", "mbh_rm_modeling", "mbh_rm_ref",
        "source_url"
    ]

    for varname in range(args.start, args.end + 1):
        tag = f"{varname:03d}"
        try:
            html = fetch_html_curl(varname, timeout=args.timeout)

            if args.save_html:
                (outdir / f"details_varname_{tag}.html").write_text(html, encoding="utf-8")

            df = extract_reverb_table_df(html)       # <-- unchanged structure
            meta = parse_metadata(html, varname)     # <-- object-level info

            # Append metadata columns to the RIGHT (preserve table structure)
            for k in meta_order:
                df[k] = meta.get(k)

            out_csv = outdir / f"details_varname_{tag}_reverb_with_meta.csv"
            df.to_csv(out_csv, index=False, encoding=args.encoding, sep=args.sep)

            ok_count += 1
            print(f"[OK] varname={varname:>3} -> {out_csv.name}  shape={df.shape}")

        except Exception as e:
            fail_count += 1
            msg = f"[FAIL] varname={varname}: {e}"
            print(msg, file=sys.stderr)
            error_log.write_text(
                (error_log.read_text(encoding="utf-8") if error_log.exists() else "")
                + msg + "\n",
                encoding="utf-8"
            )

        time.sleep(args.sleep)

    print("\nDone.")
    print(f"  Success: {ok_count}")
    print(f"  Failed : {fail_count}")
    if fail_count:
        print(f"  See: {error_log.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
