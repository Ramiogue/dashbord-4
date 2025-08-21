# app.py — Merchant Dashboard (Excel-only uploads)
# - Login by Device Serial (from Secrets)
# - Admin uploads 1..N Excel files -> APPEND/MERGE into master CSV with optional dedupe
# - Each upload is saved under data/uploads/ (keeps original name by default)
# - Rows are tagged with __source_file__ (original upload name)
# - Admin can view all merchants or choose a device
# - Merchants see only their own device
# - Upload Log: Original file | Rows | Uploaded at | Saved file | Delete
# - Only Excel is accepted (.xlsx/.xlsm/.xls/.xlsb). Master is stored as CSV.
# - BUGFIX: approved_mask uses .str.startswith("00")

import os, re, math, datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit_authenticator as stauth

# =========================
# Page
# =========================
st.set_page_config(page_title="Merchant Dashboard", page_icon=None, layout="wide")

# =========================
# Brand & Theme Tokens
# =========================
PRIMARY = "#0B6E4F"; GREEN_2 = "#149E67"; TEXT = "#0f172a"; CARD_BG = "#ffffff"
GREY_50 = "#f8fafc"; GREY_100 = "#f1f5f9"; GREY_200 = "#e2e8f0"; GREY_300 = "#cbd5e1"; GREY_400 = "#94a3b8"
GREY_700 = "#334155"
SIDEBAR_BG = "#eef2f6"; FILTER_HDR_BG_DEF = PRIMARY; FILTER_HDR_BG_OPEN = GREEN_2; FILTER_CNT_BG_OPEN = "#e8f5ef"
DANGER = "#dc2626"; NEUTRALS = ["#334155","#475569","#64748b","#94a3b8","#cbd5e1","#e2e8f0"]
LOGO_URL = "https://admin.spazaeats.co.za/public/assets/img/logo.png"

def apply_plotly_layout(fig):
    fig.update_layout(template="plotly_white", margin=dict(l=10,r=10,t=46,b=10),
                      paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                      font=dict(color=TEXT, size=12), title_x=0.01,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(showgrid=True, gridcolor=GREY_200, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GREY_200, zeroline=False)
    return fig

def currency_fmt(x):
    try: return f"R {float(x):,.0f}"
    except Exception: return "R 0"

def section_title(txt):
    return f"""<div class="section-title"><h2>{txt}</h2></div>"""

def safe_rerun():
    try: st.rerun()
    except Exception:
        try: st.experimental_rerun()
        except Exception: pass

# =========================
# Excel readers + header cleaner
# =========================
EXCEL_SHEET = st.secrets.get("EXCEL_SHEET", 0)  # sheet name or index

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
                  .str.replace('\ufeff','', regex=False)  # remove BOM
                  .str.strip()
    )
    return df

def read_excel_file(path_or_buf):
    """
    Read Excel only: .xlsx/.xlsm (openpyxl), .xls (xlrd), .xlsb (pyxlsb).
    """
    s = str(path_or_buf).lower()
    if s.endswith(".xlsx") or s.endswith(".xlsm"):
        df = pd.read_excel(path_or_buf, sheet_name=EXCEL_SHEET, engine="openpyxl")
    elif s.endswith(".xls"):
        df = pd.read_excel(path_or_buf, sheet_name=EXCEL_SHEET, engine="xlrd")
    elif s.endswith(".xlsb"):
        df = pd.read_excel(path_or_buf, sheet_name=EXCEL_SHEET, engine="pyxlsb")
    else:
        raise ValueError("Only Excel files (.xlsx, .xlsm, .xls, .xlsb) are supported.")
    return clean_cols(df)

# =========================
# Global CSS
# =========================
st.markdown(f"""
<style>
.stApp {{ background:{GREY_50}; }}
.block-container {{ padding-top:.4rem; padding-bottom:1.2rem; max-width:1320px; margin:0 auto; }}

/* Buttons: full-width, wrapped */
.stButton > button, .stDownloadButton > button {{
  width: 100%; white-space: normal; overflow-wrap: anywhere;
  background: #ffffff; color: {TEXT};
  border: 1px solid {GREY_200}; border-radius: 12px;
  box-shadow: 0 1px 2px rgba(2,6,23,0.04);
}}

.header-row {{ display:flex; align-items:center; gap:12px; margin-bottom:.25rem; }}
.header-logo img {{ height:48px; width:auto; border-radius:6px; }}
.title-left h1 {{ font-size:1.8rem; font-weight:800; margin:0; color:{TEXT}; letter-spacing:.2px; }}

.section-title h2 {{ font-size:1.3rem; margin:12px 0 6px 0; color:{TEXT}; position:relative; padding-bottom:8px; }}
.section-title h2:after {{ content:""; position:absolute; left:0; bottom:0; height:3px; width:64px; background:{PRIMARY}; border-radius:3px; }}

.card {{ background:{CARD_BG}; border:1px solid {GREY_200}; border-radius:12px;
        padding:12px; box-shadow:0 1px 2px rgba(2,6,23,0.04); margin-bottom:10px; }}
.kpi-card {{ background:{CARD_BG}; border:1px solid {GREY_200}; border-left:4px solid {PRIMARY};
            border-radius:12px; padding:10px 12px; box-shadow:0 1px 2px rgba(2,6,23,0.04);
            height:84px; display:flex; flex-direction:column; gap:2px; }}
.kpi-title {{ font-size:.72rem; color:{GREY_400}; margin:0; }}
.kpi-value {{ font-size:1.25rem; font-weight:800; color:{TEXT}; margin:0; }}
.kpi-sub {{ font-size:.75rem; color:{GREY_400}; margin:0; }}

.action-card {{
  background:{GREY_700}; color:#fff; border-radius:12px; padding:12px;
  border:1px solid {GREY_300}; box-shadow:0 1px 2px rgba(2,6,23,0.06); margin:10px 0;
}}
.action-card label, .action-card p, .action-card span, .action-card div {{ color:#fff !important; }}
.action-card .stButton > button {{ background:#ffffff; color:{TEXT}; }}

[data-testid="stSidebar"] {{ background:{SIDEBAR_BG}; box-shadow:inset -1px 0 0 {GREY_200}; }}
[data-testid="stSidebar"] details {{ border:1px solid {GREY_200}; border-radius:12px; overflow:hidden; }}
[data-testid="stSidebar"] details > summary.streamlit-expanderHeader {{
  background:{FILTER_HDR_BG_DEF} !important; color:#fff !important; font-weight:700; padding:8px 12px; }}
[data-testid="stSidebar"] details[open] > summary.streamlit-expanderHeader {{
  background:{FILTER_HDR_BG_OPEN} !important; color:#fff !important; border-bottom:1px solid {GREY_200} !important; }}
[data-testid="stSidebar"] details[open] .streamlit-expanderContent {{ background:{FILTER_CNT_BG_OPEN} !important; padding:8px 12px !important; }}

.soft-divider {{ height:10px; border-radius:999px; background:{GREY_100}; border:1px solid {GREY_200}; margin:6px 0 16px 0; }}
</style>
""", unsafe_allow_html=True)

# =========================
# Auth & Config from Secrets
# =========================
users_cfg = st.secrets.get("users", {})
cookie_key = st.secrets.get("COOKIE_KEY", "change-me")
MERCHANT_ID_COL = st.secrets.get("merchant_id_col", "Merchant Number - Business Name")
LOGIN_KEY_COL   = st.secrets.get("login_key_col", "Device Serial")
ADMIN_USERS     = set(st.secrets.get("admin_users", []))

# Upload behavior controls
UPLOAD_TIMESTAMP_PREFIX = bool(st.secrets.get("UPLOAD_TIMESTAMP_PREFIX", False))   # False = keep original names
UPLOAD_OVERWRITE        = bool(st.secrets.get("UPLOAD_OVERWRITE", True))           # True = overwrite same-name

# Build credentials
creds = {"usernames": {}}
for uname, u in users_cfg.items():
    creds["usernames"][uname] = {"name": u.get("name", uname),
                                 "email": u.get("email", ""),
                                 "password": u.get("password_hash", "")}

authenticator = stauth.Authenticate(
    credentials=creds, cookie_name="merchant_portal",
    key=cookie_key, cookie_expiry_days=7,
)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    authenticator.login(location="main")
    st.markdown('</div>', unsafe_allow_html=True)

auth_status = st.session_state.get("authentication_status")
name = st.session_state.get("name"); username = st.session_state.get("username")

if auth_status is False:
    st.error("Invalid credentials"); st.stop()
elif auth_status is None:
    st.info("Please log in."); st.stop()

authenticator.logout(location="sidebar")
st.sidebar.write(f"Hello, **{name}**")
is_admin = str(username) in ADMIN_USERS

def get_user_record(cfg: dict, uname: str):
    if uname in cfg: return cfg[uname]
    uname_cf = str(uname).casefold()
    for k, v in cfg.items():
        if str(k).casefold() == uname_cf: return v
    return None

user_rec = get_user_record(users_cfg, username)
if not user_rec and not is_admin:
    st.error("User not found in secrets."); st.stop()

# =========================
# Header under Streamlit header
# =========================
st.markdown(f'''
<div class="header-row">
  <div class="header-logo"><img src="{LOGO_URL}" alt="Spaza Eats Logo"></div>
  <div class="title-left"><h1>Merchant Dashboard</h1></div>
</div>
''', unsafe_allow_html=True)

# =========================
# ADMIN data management
# =========================
MASTER_LOCAL_PATH = "data/master_transactions.csv"   # master stays as CSV on disk
UPLOAD_DIR = "data/uploads"
UPLOAD_LOG = "data/upload_log.csv"

if is_admin:
    with st.sidebar.expander("Admin: Data Management", expanded=True):
        admin_uploads = st.file_uploader(
            "Upload one or more Excel files — they will be appended to the master",
            type=["xlsx","xls","xlsm","xlsb"],
            accept_multiple_files=True, key="admin_master_multi",
        )

        DEDUP_CANDIDATES = [
            "Device Serial","Transaction Date","Request Amount","Settle Amount",
            "Auth Code","System Trace Audit Number","Online Reference Number",
            "Retrieval Reference","UTI"
        ]
        do_dedup = st.checkbox("Deduplicate after combine", value=True)

        colA, colB = st.columns([1,1])
        with colA:
            if st.button("Append & publish master CSV", use_container_width=True, disabled=not admin_uploads):
                try:
                    os.makedirs("data", exist_ok=True)
                    os.makedirs(UPLOAD_DIR, exist_ok=True)

                    df_base = pd.read_csv(MASTER_LOCAL_PATH) if os.path.exists(MASTER_LOCAL_PATH) else pd.DataFrame()

                    new_parts, saved_names, orig_names, rows_counts = [], [], [], []
                    now = dt.datetime.now()
                    ts = now.strftime("%Y%m%d-%H%M%S")
                    uploaded_at = now.strftime("%Y-%m-%d %H:%M:%S")

                    for f in admin_uploads:
                        orig = f.name or "uploaded.xlsx"
                        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', orig)
                        dest_name = f"{ts}_{safe}" if UPLOAD_TIMESTAMP_PREFIX else safe
                        dest_path = os.path.join(UPLOAD_DIR, dest_name)

                        if os.path.exists(dest_path) and not UPLOAD_OVERWRITE:
                            base, ext = os.path.splitext(safe); k = 2
                            while os.path.exists(os.path.join(UPLOAD_DIR, f"{base}({k}){ext}")): k += 1
                            dest_name = f"{base}({k}){ext}"
                            dest_path = os.path.join(UPLOAD_DIR, dest_name)

                        with open(dest_path, "wb") as out:
                            out.write(f.getbuffer())

                        saved_names.append(dest_name); orig_names.append(orig)

                        df_i = read_excel_file(dest_path)
                        df_i["__source_file__"] = orig
                        new_parts.append(df_i)
                        rows_counts.append(len(df_i))

                    df_new = pd.concat(new_parts, ignore_index=True) if new_parts else pd.DataFrame()
                    df_out = pd.concat([df_base, df_new], ignore_index=True)

                    if do_dedup and not df_out.empty:
                        subset = [c for c in DEDUP_CANDIDATES if c in df_out.columns]
                        df_out = df_out.drop_duplicates(subset=subset, keep="last") if subset else df_out.drop_duplicates(keep="last")

                    df_out.to_csv(MASTER_LOCAL_PATH, index=False)

                    log_rows = pd.DataFrame({
                        "timestamp":[ts]*len(saved_names),
                        "uploaded_at":[uploaded_at]*len(saved_names),
                        "original_name":orig_names,
                        "rows_in_file":rows_counts,
                        "saved_name":saved_names,
                    })
                    if os.path.exists(UPLOAD_LOG):
                        old = pd.read_csv(UPLOAD_LOG)
                        log_rows = pd.concat([old, log_rows], ignore_index=True)
                    log_rows.to_csv(UPLOAD_LOG, index=False)

                    st.success(f"Appended {sum(rows_counts):,} rows from {len(saved_names)} file(s).")
                    safe_rerun()
                except Exception as e:
                    st.error(f"Failed to append/publish: {e}")

        with colB:
            if st.button("Remove local master CSV", use_container_width=True):
                try:
                    if os.path.exists(MASTER_LOCAL_PATH):
                        os.remove(MASTER_LOCAL_PATH)
                        st.success("Local master CSV removed.")
                    else:
                        st.info("No local master CSV found.")
                except Exception as e:
                    st.error(f"Failed to remove: {e}")

        # ===== Upload Log with deletion & rebuild =====
        if os.path.exists(UPLOAD_LOG):
            try:
                log_df = pd.read_csv(UPLOAD_LOG)

                if "uploaded_at" not in log_df.columns and "timestamp" in log_df.columns:
                    def ts_to_human(x):
                        try: return dt.datetime.strptime(str(x), "%Y%m%d-%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                        except Exception: return ""
                    log_df["uploaded_at"] = log_df["timestamp"].apply(ts_to_human)

                def strip_ts_prefix(name): return re.sub(r'^\d{8}-\d{6}_', '', str(name))

                display = log_df.copy()
                display["Saved file"] = display["saved_name"].apply(strip_ts_prefix)
                display = display.rename(columns={
                    "original_name":"Original file",
                    "rows_in_file":"Rows",
                    "uploaded_at":"Uploaded at",
                })

                if "Uploaded at" in display.columns and display["Uploaded at"].notna().any():
                    display = display.sort_values("Uploaded at", ascending=False)
                elif "timestamp" in display.columns:
                    display = display.sort_values("timestamp", ascending=False)

                display = display.set_index("saved_name", drop=False)
                editor_view = display[["Original file","Rows","Uploaded at","Saved file"]].copy()
                editor_view["Delete"] = False

                st.markdown(section_title("Upload Log"), unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                edited = st.data_editor(
                    editor_view,
                    key="upload_log_editor",
                    use_container_width=True, height=340,
                    hide_index=True, num_rows="fixed",
                    disabled=["Original file","Rows","Uploaded at","Saved file"],
                    column_config={
                        "Rows": st.column_config.NumberColumn("Rows", format="%d"),
                        "Uploaded at": st.column_config.TextColumn("Uploaded at"),
                        "Delete": st.column_config.CheckboxColumn("Delete", help="Check to remove this upload"),
                    },
                )
                st.markdown('</div>', unsafe_allow_html=True)

                to_delete_saved_names = edited.index[edited["Delete"] == True].tolist()

                st.markdown('<div class="action-card">', unsafe_allow_html=True)
                col_del1, col_del2 = st.columns([7,3])
                with col_del1:
                    confirm_del = st.toggle("Confirm delete", value=False)
                with col_del2:
                    delete_click = st.button(
                        "Delete selected upload(s)",
                        use_container_width=True,
                        disabled=(len(to_delete_saved_names) == 0 or not confirm_del),
                    )
                st.markdown('</div>', unsafe_allow_html=True)

                if delete_click:
                    try:
                        # remove files
                        removed = 0
                        for sname in to_delete_saved_names:
                            p = os.path.join(UPLOAD_DIR, str(sname))
                            if os.path.exists(p):
                                os.remove(p); removed += 1

                        # update log
                        new_log = log_df[~log_df["saved_name"].isin(to_delete_saved_names)].copy()
                        new_log.to_csv(UPLOAD_LOG, index=False)

                        # rebuild master from remaining uploads
                        parts = []
                        if not new_log.empty:
                            for _, r in new_log.iterrows():
                                sname = str(r.get("saved_name","")).strip()
                                fpath = os.path.join(UPLOAD_DIR, sname)
                                if os.path.exists(fpath):
                                    try:
                                        df_i = read_excel_file(fpath)
                                        orig = r.get("original_name") or strip_ts_prefix(os.path.basename(fpath))
                                        df_i["__source_file__"] = orig
                                        parts.append(df_i)
                                    except Exception as e:
                                        st.warning(f"Skipped rebuilding from '{sname}': {e}")

                        if len(parts) == 0:
                            if os.path.exists(MASTER_LOCAL_PATH):
                                os.remove(MASTER_LOCAL_PATH)
                            st.success(f"Deleted {removed} upload(s). No uploads remain, so the master CSV was removed.")
                        else:
                            rebuilt = pd.concat(parts, ignore_index=True)
                            subset = [c for c in ["Device Serial","Transaction Date","Request Amount","Settle Amount",
                                                  "Auth Code","System Trace Audit Number","Online Reference Number",
                                                  "Retrieval Reference","UTI"] if c in rebuilt.columns]
                            rebuilt = rebuilt.drop_duplicates(subset=subset, keep="last") if subset else rebuilt.drop_duplicates(keep="last")
                            rebuilt.to_csv(MASTER_LOCAL_PATH, index=False)
                            st.success(f"Deleted {removed} upload(s) and rebuilt master from {len(parts)} remaining file(s).")

                        safe_rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

            except Exception as e:
                st.warning(f"Could not render upload log: {e}")

# =========================
# Load master data (local CSV only)
# =========================
def load_master_from_local(path: str):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["__path__"] = path
        return df
    return None

tx = load_master_from_local(MASTER_LOCAL_PATH)
if tx is None:
    if is_admin:
        st.error("No master data found. Upload Excel file(s) in the Admin panel.")
    else:
        st.error("Data not available yet. Please contact the admin.")
    st.stop()

# =========================
# Validate & prep data
# =========================
required_cols = {
    LOGIN_KEY_COL, "Transaction Date","Request Amount","Settle Amount",
    "Auth Code","Decline Reason","Date Payment Extract",
    "Terminal ID","Device Serial","Product Type","Issuing Bank","BIN"
}
missing = sorted(required_cols - set(tx.columns))
if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}"); st.stop()

tx[LOGIN_KEY_COL] = tx[LOGIN_KEY_COL].astype(str).str.strip()
tx["Transaction Date"] = pd.to_datetime(tx["Transaction Date"], errors="coerce")
tx["Request Amount"]   = pd.to_numeric(tx["Request Amount"], errors="coerce")
tx["Settle Amount"]    = pd.to_numeric(tx["Settle Amount"], errors="coerce")
tx["Date Payment Extract"] = tx["Date Payment Extract"].fillna("").astype(str)
for c in ["Product Type","Issuing Bank","Decline Reason","Terminal ID","Device Serial","Auth Code"]:
    tx[c] = tx[c].fillna("").astype(str)

# =========================
# Choose scope (admin) or enforce device (merchant)
# =========================
login_value = None if is_admin else (user_rec.get("device_serial") or user_rec.get("merchant_id"))

if is_admin:
    with st.sidebar.expander("Admin view", expanded=True):
        admin_scope = st.radio("Scope", ["All merchants", "Choose device"], index=1)
        if admin_scope == "All merchants":
            f0 = tx.copy(); effective_label = "All merchants"
        else:
            devices = sorted(d for d in tx[LOGIN_KEY_COL].astype(str).str.strip().unique() if d and d.lower() != "nan")
            selected = st.selectbox("Device Serial", devices)
            f0 = tx[tx[LOGIN_KEY_COL].astype(str).str.strip() == str(selected).strip()].copy()
            effective_label = selected
else:
    if not login_value:
        st.error("Your account is not linked to a device. Contact the admin."); st.stop()
    f0 = tx[tx[LOGIN_KEY_COL].astype(str).str.strip() == str(login_value).strip()].copy()
    effective_label = login_value

if f0.empty:
    st.warning(f"No transactions for '{effective_label}' in '{LOGIN_KEY_COL}'."); st.stop()

# =========================
# Flags
# =========================
def approved_mask(df):
    dr = df["Decline Reason"].astype(str).str.strip()
    auth = df["Auth Code"].astype(str).str.strip()
    return dr.str.startswith("00") | auth.ne("")

def settled_mask(df):
    has_extract = df["Date Payment Extract"].astype(str).str.strip().ne("")
    nonzero = pd.to_numeric(df["Settle Amount"], errors="coerce").fillna(0).ne(0)
    return has_extract & nonzero

f0["is_approved"] = approved_mask(f0)
f0["is_settled"]  = settled_mask(f0)

# =========================
# Sidebar filters
# =========================
with st.sidebar.expander("Filters", expanded=True):
    valid_dates = f0["Transaction Date"].dropna()
    min_d = pd.to_datetime(valid_dates.min()).date()
    max_d = pd.to_datetime(valid_dates.max()).date()
    date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    def _multi(label, series):
        opts = sorted(series.astype(str).unique())
        return st.multiselect(label, options=opts, default=opts)

    sel_declines = _multi("Decline Reason", f0["Decline Reason"])
    sel_prodtype = _multi("Product Type",    f0["Product Type"])
    sel_issuer   = _multi("Issuing Bank",    f0["Issuing Bank"])

start_date, end_date = (date_range if isinstance(date_range, tuple) else (min_d, max_d))
flt = (f0["Transaction Date"].dt.date >= start_date) & (f0["Transaction Date"].dt.date <= end_date)
flt &= f0["Decline Reason"].isin(sel_declines)
flt &= f0["Product Type"].isin(sel_prodtype)
flt &= f0["Issuing Bank"].isin(sel_issuer)
f = f0[flt].copy()
if f.empty:
    st.warning("No data for the selected filters."); st.stop()

# =========================
# KPIs
# =========================
def safe_div(n, d): return (n / d) if d else np.nan
transactions_cnt = int(len(f))
attempts_cnt  = int(len(f))
approved_cnt  = int(f["is_approved"].sum())
approval_rate = safe_div(approved_cnt, attempts_cnt)
decline_rate  = safe_div(attempts_cnt - approved_cnt, attempts_cnt)
settled_rows = f["is_settled"]
revenue      = float(f.loc[settled_rows, "Settle Amount"].sum())
settled_cnt  = int(settled_rows.sum())
aov          = safe_div(revenue, settled_cnt)

# =========================
# Caption under header
# =========================
src_path = f"file:{MASTER_LOCAL_PATH}" if os.path.exists(MASTER_LOCAL_PATH) else "—"
st.caption(f"{LOGIN_KEY_COL}: **{effective_label}**  •  Source: `{src_path}`  •  Date: {start_date} → {end_date}")

def kpi_card(title, value, sub=""):
    st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
    """, unsafe_allow_html=True)

cols = st.columns(6, gap="small")
with cols[0]: kpi_card("# Transactions", f"{transactions_cnt:,}")
with cols[1]: kpi_card("Total Requests", currency_fmt(f['Request Amount'].sum()))
with cols[2]: kpi_card("Revenue", currency_fmt(revenue))
with cols[3]: kpi_card("Approval Rate", f"{(approval_rate*100):.1f}%" if not math.isnan(approval_rate) else "—")
with cols[4]: kpi_card("Decline Rate", f"{(decline_rate*100):.1f}%" if not math.isnan(decline_rate) else "—")
with cols[5]: kpi_card("Average Order Value (AOV)", f"R {aov:,.2f}" if not math.isnan(aov) else "—")

st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

# =========================
# Charts
# =========================
st.markdown(section_title("Revenue per Month"), unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
df_month = (
    f.loc[settled_rows, ["Transaction Date", "Settle Amount"]]
      .assign(month_start=lambda d: pd.to_datetime(d["Transaction Date"]).dt.to_period("M").dt.to_timestamp())
      .groupby("month_start", as_index=False).agg(revenue=("Settle Amount", "sum")).sort_values("month_start")
)
if not df_month.empty:
    full_months = pd.date_range(df_month["month_start"].min(), df_month["month_start"].max(), freq="MS")
    df_month = df_month.set_index("month_start").reindex(full_months, fill_value=0).rename_axis("month_start").reset_index()
    fig_m = px.line(df_month, x="month_start", y="revenue", markers=True)
    fig_m.update_traces(line=dict(color=PRIMARY, width=2), marker=dict(color=PRIMARY))
    fig_m.update_xaxes(title_text="", tickformat="%b %Y", dtick="M1")
    fig_m.update_yaxes(title_text="Revenue (R)", tickprefix="R ", separatethousands=True)
    fig_m.update_layout(title_text="Revenue per Month (Line)")
    fig_m.update_traces(hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: R %{y:,.0f}<extra></extra>")
    apply_plotly_layout(fig_m)
    st.plotly_chart(fig_m, use_container_width=True)
else:
    st.info("No settled revenue in the selected period.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(section_title("Product Type Mix (Pie)"), unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
prod_pie = f.loc[settled_rows, ["Product Type", "Settle Amount"]].copy()
if not prod_pie.empty:
    prod_pie = (prod_pie.groupby("Product Type", as_index=False)["Settle Amount"].sum()
                .rename(columns={"Settle Amount":"revenue"}).sort_values("revenue", ascending=False))
    fig_pie_pt = px.pie(prod_pie, values="revenue", names="Product Type", hole=0.5)
    fig_pie_pt.update_traces(textposition="inside",
                             texttemplate="%{label}<br>R %{value:,.0f}<br>%{percent:.0%}",
                             hovertemplate="%{label}<br>Revenue: R %{value:,.0f} (%{percent:.1%})<extra></extra>",
                             marker=dict(line=dict(color="#ffffff", width=1)))
    fig_pie_pt.update_layout(title_text="Revenue by Product Type", colorway=NEUTRALS)
    apply_plotly_layout(fig_pie_pt)
    st.plotly_chart(fig_pie_pt, use_container_width=True)
else:
    st.info("No settled revenue in the selected period.")
st.markdown('</div>', unsafe_allow_html=True)

c1, c2 = st.columns((1.2, 1), gap="small")
with c1:
    st.markdown(section_title("Top Issuing Banks by Revenue"), unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    issuer_df = (f.loc[settled_rows, ["Issuing Bank","Settle Amount"]]
                 .groupby("Issuing Bank", as_index=False).agg(revenue=("Settle Amount","sum"))
                 .sort_values("revenue", ascending=False).head(10))
    if not issuer_df.empty:
        fig_bank = px.bar(issuer_df.sort_values("revenue"), x="revenue", y="Issuing Bank", orientation="h")
        fig_bank.update_traces(marker_color=NEUTRALS[0], marker_line_color="#ffffff", marker_line_width=1)
        fig_bank.update_xaxes(title_text="Revenue (R)", tickprefix="R ", separatethousands=True)
        fig_bank.update_yaxes(title_text="")
        fig_bank.update_layout(title_text="Top 10 Issuers (Revenue)")
        fig_bank.update_traces(hovertemplate="%{y}<br>Revenue: R %{x:,.0f}<extra></extra>")
        apply_plotly_layout(fig_bank)
        st.plotly_chart(fig_bank, use_container_width=True)
    else:
        st.info("No revenue in range.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown(section_title("Top Decline Reasons"), unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    base_attempts = int(len(f))
    decl_df = (
        f.loc[~f["is_approved"], "Decline Reason"]
         .value_counts()
         .rename_axis("Decline Reason")
         .reset_index(name="count")
         .sort_values("count", ascending=True)
    )
    if not decl_df.empty and base_attempts > 0:
        decl_df["pct_of_attempts"] = decl_df["count"] / base_attempts
        fig_decl = px.bar(decl_df, x="pct_of_attempts", y="Decline Reason", orientation="h")
        fig_decl.update_traces(marker_color=DANGER, texttemplate="%{x:.0%}", textposition="outside")
        fig_decl.update_xaxes(tickformat=".0%", range=[0, max(0.01, float(decl_df["pct_of_attempts"].max()) * 1.15)])
        fig_decl.update_yaxes(title_text="")
        fig_decl.update_layout(title_text="As % of All Attempts")
        fig_decl.update_traces(hovertemplate="%{y}<br>% of Attempts: %{x:.1%}<extra></extra>")
        apply_plotly_layout(fig_decl)
        st.plotly_chart(fig_decl, use_container_width=True)
    else:
        st.info("No declines in the selected period.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Transactions table
# =========================
st.markdown(section_title("Transactions (Filtered)"), unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
show_cols = [
    "Transaction Date","Request Amount","Settle Amount",
    "Decline Reason","Auth Code","Issuing Bank","BIN","Product Type",
    "Terminal ID","Device Serial","Date Payment Extract",
    "System Batch Number","Device Batch Number","System Trace Audit Number",
    "Retrieval Reference","UTI","Online Reference Number","__source_file__"
]
existing_cols = [c for c in show_cols if c in f.columns]
tbl = f[existing_cols].sort_values("Transaction Date", ascending=False).reset_index(drop=True)
for col in ["Request Amount","Settle Amount"]:
    if col in tbl.columns:
        tbl[col] = tbl[col].apply(lambda v: f"R {v:,.2f}" if pd.notnull(v) else "")
st.dataframe(tbl, use_container_width=True, height=520)

raw_for_download = f[existing_cols].sort_values("Transaction Date", ascending=True).reset_index(drop=True)
st.download_button("Download filtered transactions (CSV)",
                   data=raw_for_download.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_transactions.csv", mime="text/csv")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footnote
# =========================
with st.expander("About the metrics"):
    st.write("""
- **# Transactions**: all rows in the selected range.
- **Approval Rate**: rows with approval (Decline Reason starts with `"00"` or Auth Code present) ÷ all rows.
- **Revenue**: sum of **Settle Amount** where a settlement file exists (`Date Payment Extract` present) and amount ≠ 0.
- **AOV**: Revenue ÷ # of settled rows.
- **Total Requests**: Sum of **Request Amount** for all rows in the selected range.
""")
