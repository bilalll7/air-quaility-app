import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import folium_static

# Load dataset
def load_data():
    df = pd.read_csv("./assets/PRSA_Data_Wanshouxigong_20130301-20170228.csv")
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    
    # Menentukan musim berdasarkan bulan
    def get_season(month):
        if month in [3, 4, 5]:
            return "Semi"
        elif month in [6, 7, 8]:
            return "Panas"
        elif month in [9, 10, 11]:
            return "Gugur"
        else:
            return "Dingin"
    
    df['season'] = df['month'].apply(get_season)
    
    # Menetapkan PM2.5 sebagai AQI proxy
    df['AQI'] = df['PM2.5']
    
    # Hapus data NaN sebelum analisis
    df = df.dropna()
    
    return df

data = load_data()

# Streamlit UI
st.title("Dashboard Kualitas Udara Beijing (Wanshouxigong )")
st.write("Dataset: PRSA Data - Wanshouxigong (2013-2017)")

# Sidebar Filters
year_selected = st.sidebar.selectbox("Pilih Tahun", sorted(data['year'].unique()))
season_selected = st.sidebar.selectbox("Pilih Musim", ['Semi', 'Panas', 'Gugur', 'Dingin'])
# Tambahan Menu untuk Anggota Kelompok
show_team = st.sidebar.checkbox("Tampilkan Anggota Kelompok")
if show_team:
    st.sidebar.subheader("Anggota Kelompok IF6-10123221:")
    team_members = [
        "10123221 - Albilal Bintang Iskandar",
        "10123241 - Muhammad Fajar Ramadhan",
        "10123243 - Farhan Habibi Hasibuan",
        "10123223 - Panji Raditya Gaib",
        "10123235 - Raffi Putra Pratama",
        "10123249 - Muhammad Mizwar Dermawan"
    ]
    for member in team_members:
        st.sidebar.write(member)

# Filter Data
data_filtered = data[(data['year'] == year_selected) & (data['season'] == season_selected)]

# Statistik Rata-rata AQI per Musim
st.subheader(f"Statistik AQI untuk Musim {season_selected} Tahun {year_selected}")
st.write("Tabel ini menunjukkan ringkasan statistik dari AQI dan polutan utama yang ada dalam dataset untuk musim dan tahun yang dipilih.")
st.write("Nilai rata-rata, standar deviasi, nilai minimum dan maksimum disajikan agar kita dapat memahami bagaimana penyebaran kualitas udara.")
st.write(data_filtered[['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].describe())

# Visualisasi Tren AQI per Tahun
st.subheader("Tren AQI dari Tahun ke Tahun")
st.write("Grafik ini menunjukkan bagaimana kualitas udara berubah selama beberapa tahun berdasarkan rata-rata AQI untuk setiap musim.")
st.write("Dapat digunakan untuk melihat apakah ada tren perbaikan atau penurunan kualitas udara dari tahun ke tahun.")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=data, x='year', y='AQI', hue='season', marker='o', ax=ax)
ax.set_xlabel("Tahun")
ax.set_ylabel("Rata-rata AQI (PM2.5)")
ax.set_title("Perubahan AQI dari Tahun ke Tahun")
st.pyplot(fig)

# Tren AQI dalam Setahun berdasarkan Musim
st.subheader(f"Perubahan AQI dalam Musim {season_selected} Tahun {year_selected}")
st.write("Grafik ini menggambarkan bagaimana rata-rata AQI berubah dalam satu musim tertentu sepanjang bulan.")
st.write("Dengan melihat pola ini, kita dapat menentukan apakah ada bulan-bulan tertentu yang memiliki kualitas udara lebih buruk atau lebih baik.")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=data_filtered, x='month', y='AQI', marker='o', ax=ax)
ax.set_xlabel("Bulan")
ax.set_ylabel("Rata-rata AQI (PM2.5)")
ax.set_title(f"Perubahan AQI selama {season_selected} Tahun {year_selected}")
st.pyplot(fig)

# Bar Chart AQI tinggi per Musim
st.subheader("Musim dengan AQI Tinggi")
st.write("Grafik batang ini menunjukkan jumlah kejadian AQI tinggi (>150) di setiap musim.")
st.write("Nilai AQI di atas 150 dianggap sebagai kondisi udara yang tidak sehat, sehingga grafik ini membantu memahami musim dengan risiko polusi tinggi.")
aqi_threshold = 150  # AQI di atas 150 dianggap buruk
high_aqi_counts = data[data['AQI'] > aqi_threshold].groupby('season').size()
fig, ax = plt.subplots()
high_aqi_counts.plot(kind='bar', color=['green', 'blue', 'orange', 'red'], ax=ax)
ax.set_ylabel("Jumlah Kejadian AQI Tinggi")
ax.set_xlabel("Musim")
ax.set_title("Frekuensi AQI Buruk per Musim")
st.pyplot(fig)

# Analisis Lanjutan - K-Means Clustering
st.subheader("Analisis Data Mining: K-Means Clustering")
st.write("Teknik K-Means digunakan untuk mengelompokkan kualitas udara berdasarkan polutan utama.")
st.write("Dengan clustering, kita dapat melihat apakah ada pola tersembunyi dalam dataset yang menunjukkan tingkat polusi yang berbeda.")

# Bersihkan data sebelum clustering
data_cleaned = data.dropna(subset=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']).copy()
features = data_cleaned[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
data_cleaned['Cluster'] = kmeans.fit_predict(features_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=data_cleaned['PM2.5'], y=data_cleaned['PM10'], hue=data_cleaned['Cluster'], palette='viridis', ax=ax)
ax.set_title("Cluster Kualitas Udara Berdasarkan PM2.5 dan PM10")
st.pyplot(fig)

# Geoanalysis: Peta dengan Folium
st.subheader("Geoanalysis: Peta Kualitas Udara")
st.write("Peta ini menunjukkan lokasi dengan AQI tinggi (merah) dan rendah (hijau).")
st.write("Setiap titik dalam peta mewakili sebuah kejadian pengukuran kualitas udara, di mana warna merah menandakan polusi tinggi dan hijau menandakan kualitas udara yang lebih baik.")
map_center = [39.9, 116.4]  # Koordinat Beijing
m = folium.Map(location=map_center, zoom_start=10)

# Hapus NaN sebelum membuat peta
data_filtered = data_filtered.dropna(subset=['PM2.5', 'PM10'])

for _, row in data_filtered.iterrows():
    folium.CircleMarker(
        location=[map_center[0] + (row['PM2.5'] * 0.0001), map_center[1] + (row['PM10'] * 0.0001)],
        radius=5,
        color='red' if row['AQI'] > 150 else 'green',
        fill=True,
        fill_color='red' if row['AQI'] > 150 else 'green',
        fill_opacity=0.7,
        popup=f"AQI: {row['AQI']}"
    ).add_to(m)

folium_static(m)

# Kesimpulan
st.subheader("Kesimpulan")
st.write("1. **Tren Kualitas Udara** - Kualitas udara di Beijing mengalami fluktuasi, dengan musim dingin cenderung memiliki AQI lebih tinggi.")
st.write("2. **Polutan Utama** - PM2.5 dan PM10 adalah polutan dominan yang berkontribusi terhadap tingginya AQI.")
st.write("3. **Pola Musiman** - Beberapa bulan memiliki lonjakan AQI yang signifikan, terutama di musim dingin.")
st.write("4. **Distribusi Polusi** - Peta menunjukkan lokasi-lokasi dengan tingkat AQI tinggi tersebar di beberapa wilayah tertentu.")
st.write("5. **Rekomendasi** - Perlu kebijakan pengendalian emisi lebih ketat dan kesadaran masyarakat untuk mengurangi aktivitas polusi.")


# Footer
st.write("Dashboard ini dibuat menggunakan Streamlit berdasarkan analisis UTS dan dilengkapi dengan teknik data mining serta geoanalysis.")