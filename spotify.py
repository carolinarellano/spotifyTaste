import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from skimage import io

# Make page full size
st.set_page_config(layout="wide")

# Custom CSS for gradient background
st.markdown(
    """
    <style>
    .stApp {
        background: #191414;
        height: 100vh;
        width: 100%;
        position: absolute;
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Configuración de estilo de fondo acorde a Spotify
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');
    .header {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 20px;
    }
    .title {
        color: green;
        text-align: center;
        font-size: 60px;
        font-family: 'Montserrat', sans-serif;
    }
    h1 {
        color: white !important;}
    .subtitles{
        color: #1DB954 !important;
        text-align: center;
        font-size: 40px;
        font-family: 'Montserrat', sans-serif;
    }
    .topic{
        color: #1DB954 !important;
        text-align: left;
        font-size: 20px;
        font-family: 'Montserrat', sans-serif;
    }
    .topic1{
        color: palegreen !important;
        text-align: left;
        font-size: 20px;
        font-family: 'Montserrat', sans-serif;
    }
    .members {
        text-align: center;
        color: white !important;
    }
    .text{
        text-align: justify;
        color: white !important;
    }
    ul {
        color: white !important;
    }
    p {
        color: white !important;
    }
    .navbar {
    position: fixed;
    top: 0;
    width: 100%;
    background-color: #1DB954;
    z-index: 10;
    text-align: center;
    padding: 10px;
    }
    .navbar a {
        color: white;
        margin: 0 15px;
        text-decoration: none;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Cargar imágenes
Logo = io.imread(r'./img/ITESO_Logo.png')
Spotify = io.imread(r'./img/Taste.png')
DecisionTree = io.imread(r'./forStreamLit/decision_tree1.png')
acuraccyDT = io.imread(r'./forStreamLit/acuraccyDTree.png')
Knn = io.imread(r'./forStreamLit/KnnModel.png')
acuraccyKnn = io.imread(r'./forStreamLit/acuraccyKnn.png')

# Centering the content
# st.markdown('<h1 class="title" style="color:black;">Spotify Taste</h1>', unsafe_allow_html=True)
cols = st.columns([1, 2, 1])
with cols[1]:
   st.image(Spotify, width=1500)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown('<h1 class="title" style="color:white; font-size:20px">Equipo</h1>',unsafe_allow_html=True)
st.markdown("""<br><p class="members">\n</p>
<p class="members">1. Yochabel Martínez Cázares 738438 ISC\n</p>
<p class="members">2. Ana Carolina Arellano Valdez 738422 ISC\n</p>
<p class="members">3. Axel Leonardo Fernández Albarran 739878 ISC\n</p><BR>""",unsafe_allow_html=True)


tabs = st.tabs(["🎵 Entendimiento del negocio", "🎧 Preparación de datos", "🎹 Modelado de datos", "📝Pruebas"])
with tabs[0]:
        # Crear columnas
    col1, col2, col3 = st.columns(3)

    # Primera columna
    with col1:
        st.markdown(
            """
            <div style="display: flex; justify-content: center; margin-top: 150px">
                <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/Kaggle_logo.png" width="400">
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Segunda columna
    with col2:
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <img src="https://cdoyle.me/content/images/size/w960/2024/01/spotify.png" width="600">
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.caption("Kaggle and Spotify for Developers")


    st.markdown('<h2 class="topic">Problema de Negocio</h2>',unsafe_allow_html=True)
    st.markdown("<p class='text'>Optimización de recomendaciones musicales.</p>",unsafe_allow_html=True)
    st.markdown("""<p class='text'>Aumentar la satisfacción del usuario en Spotify mediante la mejora del sistema de recomendaciones musicales considerando los estados de ánimo actuales de los usuarios, ofreciendo sugerencias de canciones más alineadas con sus necesidades emocionales en tiempo real.</p>
                <h2 class="topic">Problema de Minería de Datos</h2>
                <p class='text'>Desarrollar un modelo de recomendación que en base búsqueda de patrones, sugiera canciones basadas en el estado de ánimo actual del usuario, mejorando la precisión y relevancia de las recomendaciones personales musicales en la plataforma Spotify.</p>""", unsafe_allow_html=True)
    
    st.markdown("""
            <h2 class="topic" style="color: ">Entendimiento de los datos</h2>
            <h2 class="topic1">Dataset</h2>
            <p class='text'>El Dataset a utilizar es desde el archivo 278k_song_labelled.csv, lo obtuvimos de kaggle, y esta información se puede obtener directamente de Spotify for Developers, este es un registro de aproximadamente 277938 canciones, el cual contiene la información correspondiente a cada canción distribuida en 13 columnas, estos datos son:\n
            - duration (ms): duración de la canción, valores enteros
            - danceability: qué tan bailable es una canción, valores flotantes entre 0.0 y 1.0, el cual se calcula con base en el tempo, la estabilidad del ritmo y la actividad, 0.0 representando lo menos bailable y 1.0 lo más bailable. 
            - energy: qué tan energética es una canción, valores flotantes entre 0.0 y 1.0, este se calcula a partir de las características de una canción, si es rápida, ruidosa o fuerte y con estos mismos se representa la intensidad de la canción.
            - loudness: qué tan ruidosa es una canción, valores flotantes basados en los decibeles que tiene, este valor puede ir desde -60 dB hasta 0 dB.
            - spechiness: detecta la presencia de palabras habladas en la pista, valores flotantes que van desde 0.33 a 0.66
            - acousticness: este valor se representa igualmente con flotantes que van desde 0.0 a 1.0
            - instrumentalness: predice si una canción no contiene voces, en este contexto, las palabras “oh”, “ah” cuentan como instrumental. De igual manera está descrito con valores flotantes de entre 0.0 a 1.0, si una canción tiene instrumentalidad que supera el 0.5 es contado como instrumental.
            - valence: este valor representa la positividad de una canción, este valor es representado con flotantes entre 0.0 y 1.0, mientras una valencia es más baja puede considerarse como triste o depresiva, mientras que si es un valor más cercano a uno representa una canción más eufórica o alegre
            - tempo: valores flotantes, que representan las pulsaciones por minuto (PPM) de la canción, este término musical describe el ritmo de una canción.
            - spec_rate: este valor representa el muestreo espectral, se utiliza para saber si la calidad de una canción es buena o no, de igual describe este dato con flotantes, sin embargo, para este proyecto no será necesario utilizar este dato.
            - labels: contiene el género o categoría al que se le relaciona, las cuales son
                - happy (1) 
                - sad (0)
                - calm (3)
                - energetic (2).

            Estos datos son muy importantes para nuestro problema de negocio, pues a través de un análisis de características de las canciones que el usuario escuche, se podrán recomendar canciones con características similares, de manera que el usuario tendrá una mejor satisfacción al recibir recomendaciones a través de Spotify.
            </p>
            
            <h2 class="topic1">Inconsistencias</h2>
            <p class='text'>En general este Dataset está muy completo, no encontramos inconsistencias de datos, no hay datos nulos ni duplicados, lo cual nos permitirá realizar un análisis de datos bastante completo.</p>
            
            <hr>
            
            """,unsafe_allow_html=True)

with tabs[1]:
    st.markdown("""
            <h2 class="subtitles">Preparacion de los Datos</h2>
            <h2 class="topic1">Selección y Limpieza</h2>
            <p class='text'> 
                En general este Dataset está muy completo, no encontramos inconsistencias de datos, no hay datos nulos ni duplicados, lo cual nos permitirá realizar un análisis de datos bastante completo.\n
            </p>    
            <p class='text'> 
                Las técnicas que utilizamos seleccionar y limpiar los datos que queremos obtener del Dataset fueron:\n
            </p>
            <p class='text'> 
                1. Carga de datos: cargamos los dos archivos .csv y los concatenamos, para después eliminar columnas repetidas.
            </p>
            <p class='text'> 
                2. Dataframe: covertimos los datos cargados en un Dataframe para poder manipular los datos.</li>
            </p>
            <p class='text'> 
                3. Drop de datos innecesarios, como lo mencionamos en la descripción de datos, la columna spec_rate no sería relevante para el proyecto, por lo que optamos por eliminarla.</li>
            </p>
            <p class='text'> 
                4. Rename: renombramos la columna “Unnamed: 0” por “song index” para ser más descriptivos con el dato de índice que nos brinda esta columna, además, renombramos la columna “labels” por “mood”, ya que indica con índice la emoción correspondiente.</li>
            </p>
            <p class='text'> 
                5. Creación de una nueva columna: ya que queremos que nuestros datos sean más claros, creamos una nueva columna la cual dice la duración de las canciones en minutos y segundos.</li>
            </p>
            <p class='text'> 
                a.	Primero copiamos los datos de “duration (ms)” a la columna “duration (mm:ss)” convirténdolos a timedelta
            </p>
            <p class='text'> 
                b.	Después aplicamos una función lambda que permite que se muestre el tiempo en un formato de minutos y segundos.
            </p>
            <p class='text'> 
                6.	Set index: utilizamos la columna “song index” como la columna de índice para este dataframe.
            </p>
            """,unsafe_allow_html=True)
    st.markdown("""
            <h2 class="topic">Integracion y transformacion de datos</h2>
            <p class='text'> 
                        Concatenamos ambos archivos csv para obtener los datos relacionados con la uri y los labels, además de eliminar las columnas duplicadas.
        Renombramos las columnas para manipularlas de mejor manera.
        Creamos una nueva columna que contiene la duración de la canción en minutos y segundos para poder interpretarla de mejor manera.
        Mapeamos los estados de ánimo a valores numéricos para poder manipularlos de mejor manera.
        Indicamos el índice de la canción.
            </p>
            """,unsafe_allow_html=True)


    # Cargamos el Dataset
    data = pd.concat([pd.read_csv('./data/278k_song_labelled.csv'),
                    pd.read_csv('./data/278k_labelled_uri.csv')], axis=1)

    # Convertimos los datos a un DataFrame
    df = pd.DataFrame(data)
    st.markdown('<h2 class="topic1">Dataset Original</h2>',unsafe_allow_html=True)

    # Eliminamos las columnas repetidas
    df = df.loc[:,~df.columns.duplicated()]

    # Eliminamos Unnamed: 0.1
    df = df.drop(['Unnamed: 0.1'], axis=1)

    st.write(df)

    # Al no haber datos nulos, podemos continuar con la preparación de los datos
    # No utilizaremos las columnas de spec_rate y labels, por lo que las eliminamos
    df = df.drop(['spec_rate'], axis=1)

    # Renombramos columnas para manipularlas
    df = df.rename(columns={ 'Unnamed: 0': 'track index', 'uri': 'track uri', 'labels': 'mood'})

    # Creamos una nueva columna que contiene la duracion la cancion en minutos y segundos para poder interpretarla de mejor manera, sin embargo seguiremos utilizando los ms para el analisis de los datos
    df['duration (mm:ss)'] = pd.to_timedelta(df['duration (ms)'], unit='ms')
    # utilizamos una funcion lambda para que la duracion solo muestre minutos y segundos
    df['duration (mm:ss)'] = df['duration (mm:ss)'].apply(lambda x: f'{int(x.total_seconds() // 60):02d}:{int(x.total_seconds() % 60):02d}')

    # Hacemos un diccionario para mapear los estados de animo a valores numericos
    emotions_mapping = {'sad': 0, 'happy': 1, 'energetic': 2, 'calm': 3}
    # Invertimos el mapeo para poder interpretar los datos
    inverted_emotions_mapping = {v: k for k, v in emotions_mapping.items()}
    # Mapeamos los estados de animo a valores numericos
    df['mood'] = df['mood'].map(inverted_emotions_mapping)

    # indicamos el indice de la cancion
    df.set_index('track index', inplace=True)

    st.markdown("""
                ```python
                    # Cargamos el Dataset
                    data = pd.concat([pd.read_csv('./data/278k_song_labelled.csv'),
                                    pd.read_csv('./data/278k_labelled_uri.csv')], axis=1)

                    # Convertimos los datos a un DataFrame
                    df = pd.DataFrame(data)

                    # Eliminamos las columnas repetidas
                    df = df.loc[:,~df.columns.duplicated()]

                    # Eliminamos Unnamed: 0.1
                    df = df.drop(['Unnamed: 0.1'], axis=1)

                    # Al no haber datos nulos, podemos continuar con la preparación de los datos
                    # No utilizaremos las columnas de spec_rate y labels, por lo que las eliminamos
                    df = df.drop(['spec_rate'], axis=1)
                    # Renombramos columnas para manipularlas
                    df = df.rename(columns={ 'Unnamed: 0': 'track index', 'uri': 'track uri', 'labels': 'mood'})

                    # Creamos una nueva columna que contiene la duracion la cancion en minutos y segundos para poder interpretarla de mejor manera, sin embargo seguiremos utilizando los ms para el analisis de los datos
                    df['duration (mm:ss)'] = pd.to_timedelta(df['duration (ms)'], unit='ms')
                    # utilizamos una funcion lambda para que la duracion solo muestre minutos y segundos
                    df['duration (mm:ss)'] = df['duration (mm:ss)'].apply(lambda x: f'{int(x.total_seconds() // 60):02d}:{int(x.total_seconds() % 60):02d}')
                    # utilizamos la funcion lambda para indicar que mood representa el estado de animo de la cancion
                    display(df)

                    # Hacemos un diccionario para mapear los estados de animo a valores numericos
                    emotions_mapping = {'sad': 0, 'happy': 1, 'energetic': 2, 'calm': 3}
                    # Invertimos el mapeo para poder interpretar los datos
                    inverted_emotions_mapping = {v: k for k, v in emotions_mapping.items()}
                    # Mapeamos los estados de animo a valores numericos
                    df['mood'] = df['mood'].map(inverted_emotions_mapping)

                    # indicamos el indice de la cancion
                    df.set_index('track index', inplace=True)
                """)

    st.write('<h1 class="topic">Resultado del procedimiento</h1>',unsafe_allow_html=True)
    # Mostrar DataFrame
    st.write('<h1 class="topic1" >DataFrame</h1>',unsafe_allow_html=True)
    st.write(df)


    # Gráficos de características
    st.write('<h1 class="topic"> Gráficas de datos después del procesamiento</h1>', unsafe_allow_html=True)
    st.write('Para mostrar las gráficas, se ha limitado el conjunto de datos a 1000 registros')
    df_graphics = df.head(1000)

    # Visualización de características
    attributes = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    fig, axs = plt.subplots(3, 3, figsize=(10, 6))
    axs = axs.flatten()  # Aplanar la matriz de ejes para acceder más fácilmente

    colors = ['crimson', 'tomato', 'darkturquoise', 'palegreen', 'gold', 'dodgerblue', 'violet', 'cornflowerblue', 'mediumvioletred']
    for i, attribute in enumerate(attributes):
        axs[i].bar(df_graphics.index, df_graphics[attribute], color=colors[i])
        axs[i].set_title(attribute)
        axs[i].set_ylabel(attribute)
        axs[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)


    st.write('## :green[Datos clasificados por mood]')
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for mood in df['mood'].unique():
        plt.figure(figsize=(14, 8))
        for i, attribute in enumerate(attributes):
            plt.subplot(3, 3, i + 1)
            plt.hist(df[df['mood'] == mood][attribute], bins=20, color=colors[i])
            plt.title(attribute)
            plt.ylabel(attribute)
            plt.xticks(rotation=45)
        
        plt.suptitle(mood.upper(), fontsize=16)  # Título principal con el valor de 'mood'
        plt.subplots_adjust(top=0.85)  # Ajusta para que no se superpongan el título y las gráficas
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Deja espacio para el título
        st.pyplot(plt)
    
with tabs[2]:
    with st.spinner("Procesando datos, por favor espera..."):
        st.success("¡Datos procesados exitosamente!")
    st.markdown('''
                <h1 class="topic" style="color:white"; font-size:"20px">Modelado de los datos</h1>
                <p class="text"> 
                    En esta etapa, se implementaron al menos dos modelos de clasificación para abordar el problema de 
                    predicción de si a un usuario le gustará una canción basada en sus playlists. Los modelos seleccionados son
                </p>
                <h2 class="topic1" font-size:"20px">Arbol de decisiones</h2>
                <p class="text"> 
                    Un modelo de clasificación que se basa en la creación de un árbol de decisiones que se utiliza para predecir la clase 
                    de un objeto. El árbol de decisión se construye de manera recursiva, dividiendo el conjunto de datos en subconjuntos 
                    más pequeños y más homogéneos.
                </p>
                ''',unsafe_allow_html=True)
    st.image(DecisionTree, use_column_width=True, caption="Árbol de toma de decisiones")
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.image(acuraccyDT,use_column_width=True,caption="Efectividad del modelo")
        
    st.markdown('''
                <h2 class="topic1" style="text-align:center;" >Analisis de los resultados</h2>
                <p class="text" style="text-align:justify; width:50%; margin:auto; display:block;"> 
                    Estos resultados representan que el modelo es confiable para clasificar canciones según estados de ánimo, con un desempeño sólido pero que 
                    podría mejorarse para las clases más complejas como "happy" o "sad". "Calm" es la clase mejor clasificada con alta precisión y recall, mientras 
                    que "happy" y "sad" muestran algo más de error.
                </p>
                <br>
                <br>
                <h2 class="topic1" style="text-align:center;" >KNN</h2>
                <p class="text" style="text-align:justify; width:90%; margin:auto; display:block;"> 
                    El modelo KNN también se utiliza para clasificación, pero a diferencia del árbol de decisiones, KNN no tiene una estructura jerárquica; simplemente 
                    encuentra los puntos más cercanos (por proximidad) a una canción desconocida y le asigna el mood mayoritario de esos puntos cercanos.
                </p>
                <br>
                <br>

                ''',unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Mostrar la primera imagen en la primera columna
    with col1:
        st.image(Knn, use_column_width=True, caption="Modelo de KNN vecinos")

    # Mostrar la segunda imagen en la segunda columna
    with col2:
        st.markdown('<br><br>', unsafe_allow_html=True)
        st.image(acuraccyKnn, use_column_width=True, caption="Efectividad del modelo")

with tabs[3]:
    st.markdown("<h3 style='text-align: center; color: white;'>Estados de ánimo de las playlist de Caro</h3>", unsafe_allow_html=True)

    # Cargar imágenes
    graficapastel1 = io.imread(r'./img/GraficaPastel1.png')
    graficapastel2 = io.imread(r'./img/GraficaPastel2.png')
    graficapastel3 = io.imread(r'./img/GraficaPastel3.png')
    graficapastel4 = io.imread(r'./img/GraficaPastel4.png')

    # Mostrar imágenes con descripciones
    col1, col2 = st.columns(2)

    # Primera columna
    with col1:
        st.image(graficapastel1, use_column_width=True, caption="Distribución inicial: Estado de ánimo 1")
        st.image(graficapastel3, use_column_width=True, caption="Distribución ajustada: Estado de ánimo 3")

    # Segunda columna
    with col2:
        st.image(graficapastel2, use_column_width=True, caption="Distribución ajustada: Estado de ánimo 2")
        st.image(graficapastel4, use_column_width=True, caption="Distribución final: Estado de ánimo 4")

    st.markdown("<p style='text-align: center; color: white;'>Estas distribuciones muestran la variación en los estados de ánimo en diferentes etapas del análisis.</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: white;'>Estados de ánimo de las playlist de Yochi</h3>", unsafe_allow_html=True)

    # Cargar imágenes
    #graficapastel1 = io.imread(r'./img/GraficaPastel1.png')
    #graficapastel2 = io.imread(r'./img/GraficaPastel2.png')
    #graficapastel3 = io.imread(r'./img/GraficaPastel3.png')
    #graficapastel4 = io.imread(r'./img/GraficaPastel4.png')
    # Primera columna
    #with col1:
    #    st.image(graficapastel1, use_column_width=True, caption="Distribución inicial: Estado de ánimo 1")
    #    st.image(graficapastel3, use_column_width=True, caption="Distribución ajustada: Estado de ánimo 3")

    # Segunda columna
    #with col2:
    #    st.image(graficapastel2, use_column_width=True, caption="Distribución ajustada: Estado de ánimo 2")
    #    st.image(graficapastel4, use_column_width=True, caption="Distribución final: Estado de ánimo 4")