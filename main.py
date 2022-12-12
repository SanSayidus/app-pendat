import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib

st.title("PENAMBANGAN DATA")
st.write("##### Nama  : San Sayidus Solatal A`la ")
st.write("##### Nim   : 200411100032 ")
st.write("##### Kelas : Penambangan Data A ")
description, upload_data, preporcessing, modeling, implementation = st.tabs(["Description", "Upload Data", "Prepocessing", "Modeling", "Implementation"])

with description:
    st.write("""# Dataset description """)
    st.write("""###### Data setini adalah : Human Stress Detection in and through Sleep (Deteksi Stres Manusia di dalam dan melalui Tidur) """)
    st.write("""###### Sumber dataset dari kaggle : https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep?select=SaYoPillow.csv""")
    st.write("""###### Penjelasan setiap kolom : """)
    st.write("""Tingkat mendengkur - snoring rate(sr): 
    
    Mendengkur atau mengorok saat tidur menjadi hal yang dapat mengganggu kualitas tidur. Dengkuran dapat terjadi karena terhambatnya atau menyempitnya saluran napas. Makin sempit saluran napas, makin keras pula suara dengkuran yang dihasilkan. """)
    st.write("""Laju pernafasan - respiration rate(rr): 
    
    Laju pernapasan yaitu jumlah napas yang dilakukan per menitnya. Jumlah napas normal manusia dewasa per menitnya berkisar di antara 12-20 kali; namun, nilai ini merujuk pada keadaan tidak berolahraga. Saat berolahraga, jumlah napas akan meningkat dari interval 12-20. """)
    st.write("""Suhu tubuh - body temperature(t): 
    
    Untuk orang dewasa, suhu tubuh normal berkisar antara 36,1-37,2 derajat Celcius. Sedangkan untuk bayi dan anak kecil, suhu tubuh normal bisa lebih tinggi, yaitu antara 36,6-38 derajat Celcius. Suhu tubuh tinggi yang dikategorikan demam berada di atas 38 derajat Celcius dan tidak mutlak berbahaya.""") 
    st.write("""Laju pergerakan tungkai - limb movement(lm): 
    
    Ekstremitas, atau sering disebut anggota gerak, adalah perpanjangan dari anggota tubuh utama.""")
    st.write("""Kadar oksigen dalam darah - blood oxygen(bo):
    
    Kadar oksigen tinggi tekanan parsial oksigen (PaO2) diatas 120 mmHg, Kadar oksigen normal (PaO2) antara 80-100bmmHg, dan Kadar oksigen rendah (PaO2) di bawah 80 mmHg """)
    st.write("""Gerakan mata - eye movement(rem): 
    
    Gerakan bola mata diatur oleh beberapa area pada otak yaitu korteks, batang otak dan serebelum sehingga terbentuk gerak bola mata yang terintegrasi. """) 
    st.write("""Jumlah jam tidur - sleeping hours(sr): 
    
    Waktu tidur yang sesuai, agar bisa mendapatkan kualitas waktu tidur yang baik. """)
    st.write("""Detak jantung - heart rate(hr): 
    
    Detak jantung normal per menit bagi orang dewasa, termasuk yang lebih tua, adalah 50 serta 100 bpm (denyut per menit). """)
    st.write("""Tingkat stres : """)
    st.write(""" 0 - rendah/normal """)
    st.write(""" 1 – sedang rendah """)
    st.write(""" 2- sedang """)
    st.write(""" 3- sedang tinggi """)
    st.write(""" 4- tinggi """)

    st.write("""Jika Anda menggunakan kumpulan data ini atau menemukan informasi ini berkontribusi terhadap penelitian Anda, silakan kutip :

    1. L. Rachakonda, AK Bapatla, SP Mohanty, dan E. Kougianos, “SaYoPillow: Kerangka Kerja IoMT Terintegrasi-Privasi-Terintegrasi Blockchain untuk Manajemen Stres Mempertimbangkan Kebiasaan Tidur”, Transaksi IEEE pada Elektronik Konsumen (TCE), Vol. 67, No. 1, Feb 2021, hlm. 20-29.
    2. L. Rachakonda, SP Mohanty, E. Kougianos, K. Karunakaran, dan M. Ganapathiraju, “Bantal Cerdas: Perangkat Berbasis IoT untuk Deteksi Stres Mempertimbangkan Kebiasaan Tidur”, dalam Prosiding Simposium Internasional IEEE ke-4 tentang Sistem Elektronik Cerdas ( iSES), 2018, hlm. 161--166.""")

with upload_data:
    st.write("###### DATASET YANG DIGUNAKAN ")
    df = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/SaYoPillow.csv')
    st.dataframe(df)

with preporcessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    #df = df.drop(columns=["date"])
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['sl'])
    y = df['sl'].values
    df
    X
    df_min = X.min()
    df_max = X.max()
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.sl).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]],
        '3' : [dumies[2]],
        '4' : [dumies[3]],
        '5' : [dumies[4]]
    })
    st.write(labels)

with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        Snoring_Rate = st.number_input('Masukkan tingkat mendengkur : ')
        Respiration_Rate = st.number_input('Masukkan laju respirasi : ')
        Body_Temperature = st.number_input('Masukkan suhu tubuh : ')
        Limb_Movement = st.number_input('Masukkan gerakan ekstremitas : ')
        Blood_Oxygen = st.number_input('Masukkan oksigen darah : ')
        Eye_Movement = st.number_input('Masukkan gerakan mata : ')
        Sleeping_Hours = st.number_input('Masukkan jam tidur : ')
        Heart_Rate = st.number_input('Masukkan detak jantung : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                Snoring_Rate,
                Respiration_Rate,
                Body_Temperature,
                Limb_Movement,
                Blood_Oxygen,
                Eye_Movement,
                Sleeping_Hours,
                Heart_Rate
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
