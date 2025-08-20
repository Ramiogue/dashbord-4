import os
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_authenticator as stauth

# =========================
# Page
# =========================
st.set_page_config(page_title="Merchant Dashboard", page_icon=None, layout="wide")

# =========================
# Brand & Theme Tokens
# =========================
PRIMARY = "#0B6E4F"      # brand green
GREEN_2 = "#149E67"      # darker green for open/hover
TEXT    = "#0f172a"      # slate-900
CARD_BG = "#ffffff"

GREY_50  = "#f8fafc"     # page bg
GREY_100 = "#f1f5f9"
GREY_200 = "#e2e8f0"
GREY_300 = "#cbd5e1"
GREY_400 = "#94a3b8"

SIDEBAR_BG         = "#eef2f6"
FILTER_HDR_BG_DEF  = PRIMARY    # expander header (collapsed)
FILTER_HDR_BG_OPEN = GREEN_2    # expander header (open)
FILTER_CNT_BG_OPEN = "#e8f5ef"  # expander content tint

DANGER  = "#dc2626"      # declines

NEUTRALS = ["#334155","#475569","#64748b","#94a3b8","#cbd5e1","#e2e8f0"]

LOGO_URL = "https://admin.spazaeats.co.za/public/assets/img/logo.png"  # <- your logo

def apply_plotly_layout(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=46, b=10),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT, size=12),
        title_x=0.01,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GREY_200, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GREY_200, zeroline=False)
    return fig

def currency_fmt(x):
    try:
        return f"R {float(x):,.0f}"
    except Exception:
        return "R 0"

def section_title(txt):
    return f"""
    <div class="section-title">
        <h2>{txt}</h2>
    </div>
    """

# =========================
# Global CSS
# =========================
st.markdown(
    f"""
    <style>
    .stApp {{ background: {GREY_50}; }}
    .block-container {{
      padding-top:.8rem; padding-bottom:1.2rem;
      max-width:1320px; margin:0 auto;
    }}

    /* Header row with logo + big title */
    .header-row {{
      display:flex; align-items:center; gap:12px; margin-bottom:.25rem;
    }}
    .header-logo img {{
      display:block; height:48px; width:auto; border-radius:6px;
    }}
    .title-left h1 {{
      font-size:1.8rem; font-weight:800; margin:0; color:{TEXT}; letter-spacing:.2px;
    }}

    /* Section title with green underline */
    .section-title h2 {{
      font-size:1.3rem; margin:12px 0 6px 0; color:{TEXT}; position:relative; padding-bottom:8px;
    }}
    .section-title h2:after {{
      content:""; position:absolute; left:0; bottom:0; height:3px; width:64px; background:{PRIMARY}; border-radius:3px;
    }}

    /* Cards & KPIs */
    .card {{
      background:{CARD_BG}; border:1px solid {GREY_200}; border-radius:12px;
      padding:12px; box-shadow:0 1px 2px rgba(2,6,23,0.04); margin-bottom:10px;
    }}
    .kpi-card {{
      background:{CARD_BG}; border:1px solid {GREY_200}; border-left:4px solid {PRIMARY};
      border-radius:12px; padding:10px 12px; box-shadow:0 1px 2px rgba(2,6,23,0.04);
      height:84px; display:flex; flex-direction:column; justify-content:center; gap:2px;
    }}
    .kpi-title {{ font-size:.72rem; color:{GREY_400}; margin:0; }}
    .kpi-value {{ font-size:1.25rem; font-weight:800; color:{TEXT}; margin:0; }}
    .kpi-sub   {{ font-size:.75rem; color:{GREY_400}; margin:0; }}

    /* Inputs */
    div[data-testid="stTextInput"] input,
    div[data-testid="stPassword"] input,
    div[data-baseweb="input"] input {{
      background:#fff !important; color:{TEXT} !important;
      border:1px solid {GREY_300} !important; border-radius:10px !important; padding:10px 12px !important;
    }}
    div[data-baseweb="input"] input:focus {{
      border:1.5px solid {PRIMARY} !important; box-shadow:0 0 0 3px rgba(11,110,79,.10) !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ background:{SIDEBAR_BG}; box-shadow:inset -1px 0 0 {GREY_200}; }}

    /* Filters expander (force green) */
    [data-testid="stSidebar"] details {{ border:1px solid {GREY_200}; border-radius:12px; overflow:hidden; }}
    [data-testid="stSidebar"] details > summary.streamlit-expanderHeader {{
      background:{FILTER_HDR_BG_DEF} !important; color:#ffffff !important; font-weight:700; padding:8px 12px; list-style:none;
    }}
    [data-testid="stSidebar"] details[open] > summary.streamlit-expanderHeader {{
      background:{FILTER_HDR_BG_OPEN} !important; color:#ffffff !important; border-bottom:1px solid {GREY_200} !important;
    }}
    [data-testid="stSidebar"] details[open] .streamlit-expanderContent {{
      background:{FILTER_CNT_BG_OPEN} !important; padding:8px 12px !important;
    }}
    [data-testid="stSidebar"] .stDateInput input {{ background:#fff !important; border:1px solid {GREY_300} !important; }}

    .soft-divider {{
      height:10px; border-radius:999px; background:{GREY_100}; border:1px solid {GREY_200}; margin:6px 0 16px 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Auth (from Secrets)
# =========================
users_cfg = st.secrets.get("users", {})
cookie_key = st.secrets.get("COOKIE_KEY", "change-me")
MERCHANT_ID_COL = st.secrets.get("merchant_id_col", "Merchant Number - Business Name")

creds = {"usernames": {}}
for uname, u in users_cfg.items():
    creds["usernames"][uname] = {"name": u["name"], "email": u["email"], "password": u["password_hash"]}

authenticator = stauth.Authenticate(
    credentials=creds,
    cookie_name="merchant_portal",
    key=cookie_key,
    cookie_expiry_days=7,
)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    authenticator.login(location="main")
    st.markdown('</div>', unsafe_allow_html=True)

auth_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")
username = st.session_state.get("username")

if auth_status is False:
    st.error("Invalid credentials"); st.stop()
elif auth_status is None:
    st.info("Please log in."); st.stop()

authenticator.logout(location="sidebar")
st.sidebar.write(f"Hello, **{name}**")

def get_user_record(cfg: dict, uname: str):
    if uname in cfg: return cfg[uname]
    uname_cf = str(uname).casefold()
    for k, v in cfg.items():
        if str(k).casefold() == uname_cf: return v
    return None

merchant_rec = get_user_record(users_cfg, username)
if not merchant_rec or "merchant_id" not in merchant_rec:
    st.error("Merchant mapping not found for this user. Check Secrets for 'merchant_id'."); st.stop()
merchant_id_value = merchant_rec["merchant_id"]

# =========================
# Load & prep data
# =========================
@st.cache_data(ttl=60)
def load_transactions():
    for p in ("sample_merchant_transactions.csv", "data/sample_merchant_transactions.csv"):
        try:
            df = pd.read_csv(p); df["__path__"] = p; return df
        except Exception:
            pass
    raise FileNotFoundError("CSV not found. Put 'sample_merchant_transactions.csv' at repo root or under 'data/'.")

tx = load_transactions()
required_cols = {
    MERCHANT_ID_COL, "Transaction Date","Request Amount","Settle Amount",
    "Auth Code","Decline Reason","Date Payment Extract",
    "Terminal ID","Device Serial","Product Type","Issuing Bank","BIN"
}
missing = required_cols - set(tx.columns)
if missing:
    st.error(f"Missing required column(s): {', '.join(sorted(missing))}"); st.stop()

tx[MERCHANT_ID_COL] = tx[MERCHANT_ID_COL].astype(str).str.strip()
tx["Transaction Date"] = pd.to_datetime(tx["Transaction Date"], errors="coerce")
tx["Request Amount"]   = pd.to_numeric(tx["Request Amount"], errors="coerce")
tx["Settle Amount"]    = pd.to_numeric(tx["Settle Amount"], errors="coerce")
tx["Date Payment Extract"] = tx["Date Payment Extract"].fillna("").astype(str)
for c in ["Product Type","Issuing Bank","Decline Reason","Terminal ID","Device Serial","Auth Code"]:
    tx[c] = tx[c].fillna("").astype(str)

f0 = tx[tx[MERCHANT_ID_COL] == merchant_id_value].copy()
if f0.empty:
    st.warning(f"No transactions for '{merchant_id_value}' in '{MERCHANT_ID_COL}'."); st.stop()

def approved_mask(df):
    dr = df["Decline Reason"].astype(str).str.strip()
    return dr.str.startswith("00") | (df["Auth Code"].astype(str).str.strip().ne(""))

def settled_mask(df):
    has_extract = df["Date Payment Extract"].astype(str).str.strip().ne("")
    nonzero = pd.to_numeric(df["Settle Amount"], errors="coerce").fillna(0).ne(0)
    return has_extract & nonzero

f0["is_approved"] = approved_mask(f0)
f0["is_settled"]  = settled_mask(f0)

# =========================
# Sidebar filters (collapsible, green header)
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
# Header with inline logo (from URL) + title
# =========================
st.markdown(
    f'''
    <div class="header-row">
      <div class="header-logo">
        <img src="{LOGO_URL}" alt="Spaza Eats Logo">
      </div>
      <div class="title-left"><h1>Merchant Dashboard</h1></div>
    </div>
    ''',
    unsafe_allow_html=True
)

src_path = f["__path__"].iat[0] if "__path__" in f.columns and not f.empty else "—"
st.caption(f"Merchant: **{merchant_id_value}**  •  Source: `{src_path}`  •  Date: {start_date} → {end_date}")

def kpi_card(title, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
# Revenue per Month — LINE
st.markdown(section_title("Revenue per Month"), unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
df_month = (
    f.loc[settled_rows, ["Transaction Date", "Settle Amount"]]
      .assign(month_start=lambda d: pd.to_datetime(d["Transaction Date"]).dt.to_period("M").dt.to_timestamp())
      .groupby("month_start", as_index=False)
      .agg(revenue=("Settle Amount", "sum"))
      .sort_values("month_start")
)
if not df_month.empty:
    full_months = pd.date_range(df_month["month_start"].min(), df_month["month_start"].max(), freq="MS")
    df_month = df_month.set_index("month_start").reindex(full_months, fill_value=0).rename_axis("month_start").reset_index()
    df_month["month_label"] = df_month["month_start"].dt.strftime("%b %Y")

    fig_m = px.line(df_month, x="month_start", y="revenue", markers=True)
    fig_m.update_traces(line=dict(color=PRIMARY, width=2), marker=dict(color=PRIMARY))
    fig_m.update_xaxes(title_text="", tickformat="%b %Y", dtick="M1")
    fig_m.update_yaxes(title_text="Revenue (R)", tickprefix="R ", separatethousands=True)
    fig_m.update_layout(title_text="Revenue per Month (Line)")
    fig_m.update_traces(hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: R %{y:,.0f}<extra></extra>")
    apply_plotly_layout(fig_m)
    st.plotly_chart(fig_m, use_container_width=True, height=360)
else:
    st.info("No settled revenue in the selected period.")
st.markdown('</div>', unsafe_allow_html=True)

# Product Type Mix — PIE
st.markdown(section_title("Product Type Mix (Pie)"), unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
prod_pie = f.loc[settled_rows, ["Product Type", "Settle Amount"]].copy()
if not prod_pie.empty:
    prod_pie = (prod_pie.groupby("Product Type", as_index=False)["Settle Amount"]
                .sum().rename(columns={"Settle Amount":"revenue"})
                .sort_values("revenue", ascending=False))
    fig_pie_pt = px.pie(prod_pie, values="revenue", names="Product Type", hole=0.5)
    fig_pie_pt.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>R %{value:,.0f}<br>%{percent:.0%}",
        hovertemplate="%{label}<br>Revenue: R %{value:,.0f} (%{percent:.1%})<extra></extra>",
        marker=dict(line=dict(color="#ffffff", width=1))
    )
    fig_pie_pt.update_layout(title_text="Revenue by Product Type", colorway=NEUTRALS)
    apply_plotly_layout(fig_pie_pt)
    st.plotly_chart(fig_pie_pt, use_container_width=True, height=420)
else:
    st.info("No settled revenue in the selected period.")
st.markdown('</div>', unsafe_allow_html=True)

# Two-column row
c1, c2 = st.columns((1.2, 1), gap="small")
with c1:
    st.markdown(section_title("Top Issuing Banks by Revenue"), unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    issuer_df = (f.loc[settled_rows, ["Issuing Bank","Settle Amount"]]
                 .groupby("Issuing Bank", as_index=False)
                 .agg(revenue=("Settle Amount","sum"))
                 .sort_values("revenue", ascending=False)
                 .head(10))
    if not issuer_df.empty:
        fig_bank = px.bar(issuer_df.sort_values("revenue"), x="revenue", y="Issuing Bank", orientation="h")
        fig_bank.update_traces(marker_color=NEUTRALS[0], marker_line_color="#ffffff", marker_line_width=1)
        fig_bank.update_xaxes(title_text="Revenue (R)", tickprefix="R ", separatethousands=True)
        fig_bank.update_yaxes(title_text="")
        fig_bank.update_layout(title_text="Top 10 Issuers (Revenue)")
        fig_bank.update_traces(hovertemplate="%{y}<br>Revenue: R %{x:,.0f}<extra></extra>")
        apply_plotly_layout(fig_bank)
        st.plotly_chart(fig_bank, use_container_width=True, height=420)
    else:
        st.info("No revenue in range.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown(section_title("Top Decline Reasons"), unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    base_attempts = int(len(f))
    decl_df = (f.loc[~f["is_approved"], ["Decline Reason"]]
               .value_counts().reset_index(name="count")
               .rename(columns={"index":"Decline Reason"})
               .sort_values("count", ascending=True))
    if not decl_df.empty and base_attempts > 0:
        decl_df["pct_of_attempts"] = decl_df["count"] / base_attempts
        fig_decl = px.bar(decl_df, x="pct_of_attempts", y="Decline Reason", orientation="h")
        fig_decl.update_traces(marker_color=DANGER, texttemplate="%{x:.0%}", textposition="outside")
        fig_decl.update_xaxes(tickformat=".0%", range=[0, max(0.01, float(decl_df["pct_of_attempts"].max()) * 1.15)])
        fig_decl.update_yaxes(title_text="")
        fig_decl.update_layout(title_text="As % of All Attempts")
        fig_decl.update_traces(hovertemplate="%{y}<br>% of Attempts: %{x:.1%}<extra></extra>")
        apply_plotly_layout(fig_decl)
        st.plotly_chart(fig_decl, use_container_width=True, height=420)
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
    "Retrieval Reference","UTI","Online Reference Number",
]
existing_cols = [c for c in show_cols if c in f.columns]
tbl = f[existing_cols].sort_values("Transaction Date", ascending=False).reset_index(drop=True)

for col in ["Request Amount","Settle Amount"]:
    if col in tbl.columns:
        tbl[col] = tbl[col].apply(lambda v: f"R {v:,.2f}" if pd.notnull(v) else "")

st.dataframe(tbl, use_container_width=True, height=520)

@st.cache_data
def to_csv_bytes():
    raw = f[existing_cols].sort_values("Transaction Date", ascending=False).reset_index(drop=True)
    return raw.to_csv(index=False).encode("utf-8")

st.download_button("Download filtered transactions (CSV)",
                   data=to_csv_bytes(), file_name="filtered_transactions.csv", mime="text/csv")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footnote
# =========================
with st.expander("About the metrics"):
    st.write(
        """
- **# Transactions**: all rows in the selected range.
- **Approval Rate**: rows with approval (Decline Reason starts with `"00"` or Auth Code present) ÷ all rows.
- **Revenue**: sum of **Settle Amount** where a settlement file exists (`Date Payment Extract` present) and amount ≠ 0.
- **AOV**: Revenue ÷ # of settled rows.
- **Total Requests**: Sum of **Request Amount** for all rows in the selected range.
"""
    )
