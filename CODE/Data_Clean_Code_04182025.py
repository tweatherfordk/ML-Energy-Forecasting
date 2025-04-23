import re
import glob
import pandas as pd

# 1) Pattern to pick up all your EIA files
WORKBOOK_PATHS = glob.glob(
    r"EIA923_Schedules_2_3_4_5*.xlsx"
)

SHEET_KEYWORD = "Page 4 Generator Data"

DESIRED_COLS = [
    "Plant Id", "Combined Heat And Power Plant", "Plant Name", "Operator Name",
    "Operator Id", "Plant State", "Census Region", "NERC Region", "NAICS Code",
    "Sector Number", "Sector Name", "Generator Id", "Reported\nPrime Mover",
    "Net Generation\nJanuary", "Net Generation\nFebruary", "Net Generation\nMarch",
    "Net Generation\nApril", "Net Generation\nMay", "Net Generation\nJune",
    "Net Generation\nJuly", "Net Generation\nAugust", "Net Generation\nSeptember",
    "Net Generation\nOctober", "Net Generation\nNovember",
    "Net Generation\nDecember", "Net Generation\nYear To Date",
    "BA_CODE", "YEAR",
]

# Which of those are “net generation” so we can coerce non-numeric → 0
NET_GEN_COLS = [c for c in DESIRED_COLS if c.startswith("Net Generation")]

# Map cleaned keys → your exact column names
COLUMN_MAPPING = {
    "plant id":                       "Plant Id",
    "combined heat and power plant": "Combined Heat And Power Plant",
    "plant name":                     "Plant Name",
    "operator name":                  "Operator Name",
    "operator id":                    "Operator Id",
    "plant state":                    "Plant State",
    "census region":                  "Census Region",
    "nerc region":                    "NERC Region",
    "naics code":                     "NAICS Code",
    "sector number":                  "Sector Number",
    "sector name":                    "Sector Name",
    "generator id":                   "Generator Id",
    "reported prime mover":           "Reported\nPrime Mover",
    "net generation january":         "Net Generation\nJanuary",
    "net generation february":        "Net Generation\nFebruary",
    "net generation march":           "Net Generation\nMarch",
    "net generation april":           "Net Generation\nApril",
    "net generation may":             "Net Generation\nMay",
    "net generation june":            "Net Generation\nJune",
    "net generation july":            "Net Generation\nJuly",
    "net generation august":          "Net Generation\nAugust",
    "net generation september":       "Net Generation\nSeptember",
    "net generation october":         "Net Generation\nOctober",
    "net generation november":        "Net Generation\nNovember",
    "net generation december":        "Net Generation\nDecember",
    "net generation year to date":    "Net Generation\nYear To Date",
    "balancing authority code":       "BA_CODE",
    "ba code":                        "BA_CODE",
    "ba_code":                        "BA_CODE",
    "year":                           "YEAR",
}

def find_header_row(xls: pd.ExcelFile, sheet: str, look_for="plant id", max_rows=15) -> int:
    """
    Read the first `max_rows` rows with no header and return the row index
    where any cell == `look_for` (case-insensitive).
    """
    preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=max_rows, dtype=str)
    for idx, row in preview.iterrows():
        if any(isinstance(cell, str) and cell.strip().lower() == look_for for cell in row):
            return idx
    raise ValueError(f"Header row not found (searched first {max_rows} rows) in sheet {sheet}")

def process_workbook(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    per_file = []

    for sheet in xls.sheet_names:
        if SHEET_KEYWORD not in sheet:
            continue

        # 2) detect exactly which row is the header
        hdr = find_header_row(xls, sheet)

        # 3) read with that row as header
        df = pd.read_excel(xls, sheet_name=sheet, header=hdr, dtype=object)

        # 4) normalize raw column names:
        #    - collapse any whitespace (spaces, newlines) → single space
        #    - lowercase & strip
        clean_cols = (
            pd.Index(df.columns.astype(str))
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
              .str.lower()
        )
        df.columns = clean_cols

        # 5) rename to your canonical names
        df = df.rename(columns=COLUMN_MAPPING)

        # 6) pull the YEAR out of sheet name or filename
        m = re.search(r"(20\d{2})", sheet) or re.search(r"(20\d{2})", path)
        df["YEAR"] = int(m.group(1)) if m else pd.NA

        # 7) coerce Net Generation cols to numeric, fill bad → 0
        for col in NET_GEN_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # 8) ensure every desired column exists
        for col in DESIRED_COLS:
            if col not in df.columns:
                df[col] = pd.NA

        # 9) select exactly in your order
        per_file.append(df[DESIRED_COLS])

    if not per_file:
        return pd.DataFrame(columns=DESIRED_COLS)

    return pd.concat(per_file, ignore_index=True)

if __name__ == "__main__":
    all_parts = []
    for wb in WORKBOOK_PATHS:
        print("→ Processing", wb)
        part = process_workbook(wb)
        if not part.empty:
            all_parts.append(part)

    if not all_parts:
        raise RuntimeError("No data found in any matching sheet!")

    consolidated = pd.concat(all_parts, ignore_index=True)

    # Write both Excel and CSV with your requested name
    out_xlsx = "2015_2024_Elec_Net_Gen_Data.xlsx"
    out_csv  = "2015_2024_Elec_Net_Gen_Data.csv"

    consolidated.to_excel(out_xlsx, index=False)
    consolidated.to_csv (out_csv,  index=False)

    print(f"✅ Done: {len(consolidated)} rows written to:")
    print(f"   • {out_xlsx}")
    print(f"   • {out_csv}")
