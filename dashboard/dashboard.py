import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium

# ------------- CONFIG -----------------
st.set_page_config(
    page_title="E‑Commerce Data Dashboard",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Dashboard Analisis E‑Commerce")
st.caption("**Proyek Analisis Data**")

# ------------- DATA LOADING -----------
@st.cache_data
def load_datasets():
    orders = pd.read_csv("../data/orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    payments = pd.read_csv("../data/order_payments_dataset.csv")
    reviews = pd.read_csv("../data/order_reviews_dataset.csv")
    geolocation = pd.read_csv("../data/geolocation_dataset.csv")
    return orders, payments, reviews, geolocation

orders, payments, reviews, geolocation = load_datasets()

# --------------------------------------
def label_seg(row):
    if row.RFM_Score >= 12: return "Loyal Customer"
    if row.RFM_Score >= 9: return "Active Customer"
    if row.R_Score == 5: return "New Customers"
    if row.RFM_Score <= 5: return "At Risk"
    return "Need Attention"

def qcut_safe(series, q=5, reverse=False):
    if series.nunique() == 1:
        return pd.Series([q // 2 + 1] * len(series), index=series.index, dtype=int)
    cats = pd.qcut(series.rank(method="first"), q=q, duplicates="drop", labels=False)
    cats = cats.astype(int) + 1
    if reverse:
        cats = cats.max() - cats + 1
    return cats

# =================================================================================
tab_overview, tab_orders, tab_reviews, tab_payments, tab_rfm, tab_geospatial = st.tabs(
    ["Overview", "Orders", "Reviews", "Payments", "RFM Segmentation", "Geospatial"]
)

# ---------- OVERVIEW TAB --------------------------------------------------------
with tab_overview:
    st.subheader("Statistik Deskriptif Singkat")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pesanan", f"{orders['order_id'].nunique():,}")
    col2.metric("Total Pelanggan", f"{orders['customer_id'].nunique():,}")
    col3.metric("Total Pembayaran", f"${payments['payment_value'].sum():,.0f}")

    st.write("### Statistik Deskriptif Data Pesanan")
    st.dataframe(orders.describe(include='all').T)

    st.write("### Statistik Deskriptif Data Pembayaran")
    st.dataframe(payments.describe(include='all').T)

    st.write("### Statistik Deskriptif Data Ulasan")
    st.dataframe(reviews.describe(include='all').T)

# ---------- ORDERS TAB ----------------------------------------------------------
with tab_orders:
    st.subheader("Jumlah Pesanan per Bulan")
    orders['bulan_pembelian'] = orders['order_purchase_timestamp'].dt.to_period("M").astype(str)
    order_per_month = orders.groupby('bulan_pembelian')['order_id'].count().reset_index()

    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(data=order_per_month, x='bulan_pembelian', y='order_id', ax=ax)
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Jumlah Pesanan")
    ax.set_title("Jumlah Pesanan per Bulan")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig, use_container_width=True)

# ---------- REVIEWS TAB ---------------------------------------------------------
with tab_reviews:
    st.subheader("Distribusi Skor Review")
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.countplot(x="review_score", data=reviews, ax=ax1)
    ax1.set_xlabel("Skor Review")
    ax1.set_ylabel("Jumlah")
    st.pyplot(fig1, use_container_width=True)

    st.subheader("Rata-rata Skor Review per Bulan")
    orders_reviews = orders[['order_id','order_purchase_timestamp']].merge(
        reviews[['order_id','review_score']], on='order_id', how='inner')
    orders_reviews['bulan_pembelian'] = orders_reviews['order_purchase_timestamp'].dt.to_period('M').astype(str)
    mean_review = orders_reviews.groupby('bulan_pembelian')['review_score'].mean().reset_index()

    fig2, ax2 = plt.subplots(figsize=(12,6))
    sns.lineplot(data=mean_review, x='bulan_pembelian', y='review_score', marker='o', ax=ax2)
    ax2.set_xlabel("Bulan")
    ax2.set_ylabel("Skor Review Rata-rata")
    ax2.set_title("Rata-rata Skor Review per Bulan")
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2, use_container_width=True)

# ---------- PAYMENTS TAB --------------------------------------------------------
with tab_payments:
    st.subheader("Distribusi Jenis Pembayaran")
    fig3, ax3 = plt.subplots(figsize=(8,5))
    counts = payments['payment_type'].value_counts().reset_index(name='count')
    counts.columns = ['payment_type', 'count']
    sns.barplot(data=counts, x='payment_type', y='count', ax=ax3)
    ax3.set_xlabel("Jenis Pembayaran")
    ax3.set_ylabel("Jumlah")
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3, use_container_width=True)

# ---------- RFM TAB -------------------------------------------------------------
with tab_rfm:
    st.subheader("Segmentasi Pelanggan (RFM)")
    rfm_raw = orders[['order_id','customer_id','order_purchase_timestamp']]\
        .merge(payments[['order_id','payment_value']], on='order_id', how='left')
    snapshot_date = rfm_raw['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

    rfm = (
        rfm_raw.groupby('customer_id')
               .agg(
                   Recency=('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
                   Frequency=('order_id', 'nunique'),
                   Monetary=('payment_value', 'sum')
               )
               .reset_index()
    )

    rfm['R_Score'] = qcut_safe(rfm['Recency'],  q=5, reverse=True)
    rfm['F_Score'] = qcut_safe(rfm['Frequency'], q=5)
    rfm['M_Score'] = qcut_safe(rfm['Monetary'],  q=5)
    rfm['RFM_Score'] = rfm[['R_Score','F_Score','M_Score']].sum(axis=1)
    rfm['Segment'] = rfm.apply(label_seg, axis=1)

    seg_counts = (
        rfm.groupby('Segment')
           .size()
           .reset_index(name='count')
           .sort_values(by='count', ascending=False)
    )

    fig4, ax4 = plt.subplots(figsize=(8,4))
    sns.barplot(data=seg_counts, y='Segment', x='count',
                order=seg_counts['Segment'].tolist(), ax=ax4)
    ax4.set_xlabel("Jumlah Pelanggan")
    ax4.set_ylabel("")
    ax4.set_title("Distribusi Segmen Pelanggan (RFM)")
    st.pyplot(fig4, use_container_width=True)

    st.write("### Tabel RFM (Top 5)")
    st.dataframe(rfm.head())

# ---------- GEOSPATIAL TAB ------------------------------------------------------
with tab_geospatial:
    st.subheader("Peta Persebaran Lokasi Pelanggan")

    lokasi_kota = geolocation.groupby("geolocation_city").agg({
        "geolocation_lat": "mean",
        "geolocation_lng": "mean"
    }).reset_index()

    peta = folium.Map(location=[-14.2350, -51.9253], zoom_start=4, tiles="CartoDB positron")

    for _, row in lokasi_kota.iterrows():
        folium.CircleMarker(
            location=[row["geolocation_lat"], row["geolocation_lng"]],
            radius=3,
            popup=row["geolocation_city"],
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.5
        ).add_to(peta)

    st_folium(peta, width=700, height=500)

    st.write("### Top‑10 Kota berdasarkan Jumlah Entri Lokasi")
    top_kota = geolocation["geolocation_city"].value_counts().head(10)
    fig5, ax5 = plt.subplots(figsize=(8,4))
    sns.barplot(x=top_kota.values, y=top_kota.index, ax=ax5, color="skyblue")
    ax5.set_title("Top‑10 Kota berdasarkan Jumlah Entri Lokasi")
    ax5.set_xlabel("Jumlah Entri")
    ax5.set_ylabel("Nama Kota")
    st.pyplot(fig5, use_container_width=True)
