import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Dashboard Data E-Commerce",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Dashboard Analisis E‑Commerce")
st.caption("**Proyek Analisis Data**")

@st.cache_data
def muat_dataset():
    pesanan = pd.read_csv("dashboard/orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    pembayaran = pd.read_csv("dashboard/order_payments_dataset.csv")
    ulasan = pd.read_csv("dashboard/order_reviews_dataset.csv")
    lokasi = pd.read_csv("dashboard/geolocation_dataset.csv")
    return pesanan, pembayaran, ulasan, lokasi

pesanan, pembayaran, ulasan, lokasi = muat_dataset()

def tentukan_segmen(baris):
    if baris.Skor_RFM >= 12: return "Pelanggan Loyal"
    if baris.Skor_RFM >= 9: return "Pelanggan Aktif"
    if baris.Skor_R == 5: return "Pelanggan Baru"
    if baris.Skor_RFM <= 5: return "Berisiko"
    return "Butuh Perhatian"

def potong_kuantil(seri, q=5, mundur=False):
    if seri.nunique() == 1:
        return pd.Series([q // 2 + 1] * len(seri), index=seri.index, dtype=int)
    hasil = pd.qcut(seri.rank(method="first"), q=q, duplicates="drop", labels=False)
    hasil = hasil.astype(int) + 1
    if mundur:
        hasil = hasil.max() - hasil + 1
    return hasil

tab_ringkasan, tab_pesanan, tab_ulasan, tab_pembayaran, tab_rfm, tab_peta = st.tabs(
    ["Ringkasan", "Pesanan", "Ulasan", "Pembayaran", "Segmentasi RFM", "Peta Lokasi"]
)

# TAB RINGKASAN
with tab_ringkasan:
    st.subheader("Statistik Deskriptif Singkat")
    kol1, kol2, kol3 = st.columns(3)
    kol1.metric("Total Pesanan", f"{pesanan['order_id'].nunique():,}")
    kol2.metric("Total Pelanggan", f"{pesanan['customer_id'].nunique():,}")
    kol3.metric("Total Pembayaran", f"${pembayaran['payment_value'].sum():,.0f}")

    st.write("### Statistik Data Pesanan")
    st.dataframe(pesanan.describe(include='all').T)

    st.write("### Statistik Data Pembayaran")
    st.dataframe(pembayaran.describe(include='all').T)

    st.write("### Statistik Data Ulasan")
    st.dataframe(ulasan.describe(include='all').T)

# TAB PESANAN
with tab_pesanan:
    st.subheader("Jumlah Pesanan per Bulan")
    pesanan['bulan'] = pesanan['order_purchase_timestamp'].dt.to_period("M").astype(str)
    jumlah_pesanan = pesanan.groupby('bulan')['order_id'].count().reset_index()

    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(data=jumlah_pesanan, x='bulan', y='order_id', ax=ax)
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Jumlah Pesanan")
    ax.set_title("Jumlah Pesanan per Bulan")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig, use_container_width=True)

# TAB ULASAN
with tab_ulasan:
    st.subheader("Distribusi Skor Ulasan")
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.countplot(x="review_score", data=ulasan, ax=ax1)
    ax1.set_xlabel("Skor Ulasan")
    ax1.set_ylabel("Jumlah")
    st.pyplot(fig1, use_container_width=True)

    st.subheader("Rata-rata Skor Ulasan per Bulan")
    gabungan = pesanan[['order_id','order_purchase_timestamp']].merge(
        ulasan[['order_id','review_score']], on='order_id', how='inner')
    gabungan['bulan'] = gabungan['order_purchase_timestamp'].dt.to_period('M').astype(str)
    rata_rata = gabungan.groupby('bulan')['review_score'].mean().reset_index()

    fig2, ax2 = plt.subplots(figsize=(12,6))
    sns.lineplot(data=rata_rata, x='bulan', y='review_score', marker='o', ax=ax2)
    ax2.set_xlabel("Bulan")
    ax2.set_ylabel("Skor Rata-rata")
    ax2.set_title("Skor Ulasan Rata-rata per Bulan")
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2, use_container_width=True)

# TAB PEMBAYARAN
with tab_pembayaran:
    st.subheader("Distribusi Metode Pembayaran")
    hitung = pembayaran['payment_type'].value_counts().reset_index(name='jumlah')
    hitung.columns = ['metode', 'jumlah']
    fig3, ax3 = plt.subplots(figsize=(8,5))
    sns.barplot(data=hitung, x='metode', y='jumlah', ax=ax3)
    ax3.set_xlabel("Metode Pembayaran")
    ax3.set_ylabel("Jumlah")
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3, use_container_width=True)

# TAB RFM
with tab_rfm:
    st.subheader("Segmentasi Pelanggan (RFM)")
    data_rfm = pesanan[['order_id','customer_id','order_purchase_timestamp']]\
        .merge(pembayaran[['order_id','payment_value']], on='order_id', how='left')
    tanggal_snapshot = data_rfm['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

    rfm = (
        data_rfm.groupby('customer_id')
                .agg(
                    Recency=('order_purchase_timestamp', lambda x: (tanggal_snapshot - x.max()).days),
                    Frequency=('order_id', 'nunique'),
                    Monetary=('payment_value', 'sum')
                )
                .reset_index()
    )

    rfm['Skor_R'] = potong_kuantil(rfm['Recency'],  q=5, mundur=True)
    rfm['Skor_F'] = potong_kuantil(rfm['Frequency'], q=5)
    rfm['Skor_M'] = potong_kuantil(rfm['Monetary'],  q=5)
    rfm['Skor_RFM'] = rfm[['Skor_R','Skor_F','Skor_M']].sum(axis=1)
    rfm['Segmen'] = rfm.apply(tentukan_segmen, axis=1)

    jumlah_segmen = (
        rfm.groupby('Segmen')
           .size()
           .reset_index(name='jumlah')
           .sort_values(by='jumlah', ascending=False)
    )

    fig4, ax4 = plt.subplots(figsize=(8,4))
    sns.barplot(data=jumlah_segmen, y='Segmen', x='jumlah',
                order=jumlah_segmen['Segmen'].tolist(), ax=ax4)
    ax4.set_xlabel("Jumlah Pelanggan")
    ax4.set_ylabel("")
    ax4.set_title("Distribusi Segmen Pelanggan")
    st.pyplot(fig4, use_container_width=True)

    st.write("### Tabel RFM (Top 5)")
    st.dataframe(rfm.head())

# TAB PETA
with tab_peta:
    st.subheader("Peta Persebaran Lokasi Pelanggan")

    lokasi_kota = lokasi.groupby("geolocation_city").agg({
        "geolocation_lat": "mean",
        "geolocation_lng": "mean"
    }).reset_index()

    peta = folium.Map(location=[-14.2350, -51.9253], zoom_start=4, tiles="CartoDB positron")

    for _, baris in lokasi_kota.iterrows():
        folium.CircleMarker(
            location=[baris["geolocation_lat"], baris["geolocation_lng"]],
            radius=3,
            popup=baris["geolocation_city"],
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.5
        ).add_to(peta)

    st_folium(peta, width=700, height=500)

    st.write("### Top‑10 Kota berdasarkan Jumlah Entri Lokasi")
    top_kota = lokasi["geolocation_city"].value_counts().head(10)
    fig5, ax5 = plt.subplots(figsize=(8,4))
    sns.barplot(x=top_kota.values, y=top_kota.index, ax=ax5, color="skyblue")
    ax5.set_title("Top‑10 Kota berdasarkan Jumlah Entri")
    ax5.set_xlabel("Jumlah Entri")
    ax5.set_ylabel("Nama Kota")
    st.pyplot(fig5, use_container_width=True)
