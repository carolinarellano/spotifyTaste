import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from skimage import io

# Cargar imágenes
Logo = io.imread(r'./img/ITESO_Logo.png')
Spotify = io.imread(r'./img/Spotify_logo.png')

# Mostrar imágenes
st.image(Logo, width=200)
st.title('Primer entrega - Proyecto Final')
st.subheader('Programación para minería de datos', )


col1, col2 = st.columns(2)

with col1:
    st.write('**Preparación y modelado de Datos**')
    st.write('Análisis de Características de Canciones de Spotify')
with col2:
    st.image(Spotify, width=200)


st.write(":red[Equipo]")
st.write("1. **Yochabel Martínez Cázares** 738438 ISC")
st.write("2. **Ana Carolina Arellano Valdez** 738422 ISC")
st.write("3. **Axel Leonardo Fernández Albarran** 739878 ISC")
st.write("4. **Claudio Bayro Jablonksi 731886 ISC**")


st.subheader(":blue[Entendimiento del Negocio]")
st.write("#### :green[Problema de Negocio]")
st.write("Optimización de recomendaciones musicales.")
st.write("Aumentar la satisfacción del usuario en Spotify mediante la mejora del sistema de recomendaciones musicales considerando los estados de ánimo actuales de los usuarios, ofreciendo sugerencias de canciones más alineadas con sus necesidades emocionales en tiempo real.")
st.write("#### :green[Problema de Minería de Datos]")
st.write("Desarrollar un modelo de recomendación que en base búsqueda de patrones, sugiera canciones basadas en el estado de ánimo actual del usuario, mejorando la precisión y relevancia de las recomendaciones personales musicales en la plataforma Spotify.")

st.subheader(":blue[Entendimiento de los Datos]")
st.write("#### :green[Dataset]")
st.write("""El Dataset a utilizar es desde el archivo 278k_song_labelled.csv, lo obtuvimos de kaggle, y esta información se puede obtener directamente de Spotify for Developers, este es un registro de aproximadamente 277938 canciones, el cual contiene la información correspondiente a cada canción distribuida en 13 columnas, estos datos son:
- **duration (ms)**: duración de la canción, valores enteros
- **danceability**: qué tan bailable es una canción, valores flotantes entre 0.0 y 1.0, el cual se calcula con base en el tempo, la estabilidad del ritmo y la actividad, 0.0 representando lo menos bailable y 1.0 lo más bailable. 
- **energy**: qué tan energética es una canción, valores flotantes entre 0.0 y 1.0, este se calcula a partir de las características de una canción, si es rápida, ruidosa o fuerte y con estos mismos se representa la intensidad de la canción.
- **loudness**: qué tan ruidosa es una canción, valores flotantes basados en los decibeles que tiene, este valor puede ir desde -60 dB hasta 0 dB.
- **spechiness**: detecta la presencia de palabras habladas en la pista, valores flotantes que van desde 0.33 a 0.66
- **acousticness**: este valor se representa igualmente con flotantes que van desde 0.0 a 1.0
- **instrumentalness**: predice si una canción no contiene voces, en este contexto, las palabras “oh”, “ah” cuentan como instrumental. De igual manera está descrito con valores flotantes de entre 0.0 a 1.0, si una canción tiene instrumentalidad que supera el 0.5 es contado como instrumental.
- **valence**: este valor representa la positividad de una canción, este valor es representado con flotantes entre 0.0 y 1.0, mientras una valencia es más baja puede considerarse como triste o depresiva, mientras que si es un valor más cercano a uno representa una canción más eufórica o alegre
- **tempo**: valores flotantes, que representan las pulsaciones por minuto (PPM) de la canción, este término musical describe el ritmo de una canción.
- **spec_rate**: este valor representa el muestreo espectral, se utiliza para saber si la calidad de una canción es buena o no, de igual describe este dato con flotantes, sin embargo, para este proyecto no será necesario utilizar este dato.
- **labels**: contiene el número de etiquetas asociadas a esta canción, las cuales se representan con enteros, generalmente son categorías como géneros, sin embargo, pueden ser clasificadas por el artista o por su sello discográfico, pero, así como con el spec_rate, no será necesario para nuestro proyecto.
Estos datos son muy importantes para nuestro problema de negocio, pues a través de un análisis de características de las canciones que el usuario escuche, se podrán recomendar canciones con características similares, de manera que el usuario tendrá una mejor satisfacción al recibir recomendaciones a través de Spotify.
""")
st.write("#### :green[Inconsistencias]")
st.write("En general este Dataset está muy completo, no encontramos inconsistencias de datos, no hay datos nulos ni duplicados, lo cual nos permitirá realizar un análisis de datos bastante completo.")

st.subheader(":blue[Preparacion de los Datos]")
st.write("#### :green[Selección y Limpieza]")
st.write("""
En general este Dataset está muy completo, no encontramos inconsistencias de datos, no hay datos nulos ni duplicados, lo cual nos permitirá realizar un análisis de datos bastante completo.
Las técnicas que utilizamos seleccionar y limpiar los datos que queremos obtener del Dataset fueron:
1.	Carga de datos: al ser sólo un archivo .csv fue bastante sencillo cargar los datos.
2.	Dataframe: covertimos los datos cargados en un Dataframe para poder manipular los datos.
3.	Drop de datos innecesarios, como lo mencionamos en la descripción de datos, las columnas spec_rate y labels no serían relevantes para el proyecto, por lo que optamos por eliminarlas
4.	Rename: renombramos la columna “Unnamed: 0” por “song index” para ser más descriptivos con el dato de índice que nos brinda esta columna.
5.	Creación de una nueva columna: ya que queremos que nuestros datos sean más claros, creamos una nueva columna la cual dice la duración de las canciones en minutos y segundos.
a.	Primero copiamos los datos de “duration (ms)” a la columna “duration (mm:ss)” convirténdolos a timedelta
b.	Después aplicamos una función lambda que permite que se muestre el tiempo en un formato de minutos y segundos.
6.	Set index: utilizamos la columna “song index” como la columna de índice para este dataframe.
""")

st.write("#### :green[Integración y Transformación de los Datos]")
st.write("""
    Para la integración y transformación de los datos, no fue necesario realizar ninguna acción, ya que el dataset se encuentra completo y no tiene datos nulos, por lo que no fue necesario realizar ninguna transformación de datos.
         """)

# Cargar el Dataset
data = pd.read_csv('./data/278k_song_labelled.csv')
df = pd.DataFrame(data)

# Preprocesamiento
df = df.drop(['spec_rate', 'labels'], axis=1)
df = df.rename(columns={'Unnamed: 0': 'song index'})
df['duration (mm:ss)'] = pd.to_timedelta(df['duration (ms)'], unit='ms')
df['duration (mm:ss)'] = df['duration (mm:ss)'].apply(lambda x: f'{int(x.total_seconds() // 60):02d}:{int(x.total_seconds() % 60):02d}')
df.set_index('song index', inplace=True)

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

st.write("#### :violet[Resultado del Procesamiento]")
# Mostrar DataFrame
st.write('## DataFrame')
st.write(df)


# Gráficos de características
st.write('## :orange[Gráficas de datos después del procesamiento]')
st.write('Para mostrar las gráficas, se ha limitado el conjunto de datos a 1000 registros')
df_graphics = df.head(1000)

# Visualización de características
attributes = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

fig, axs = plt.subplots(3, 3, figsize=(14, 8))
axs = axs.flatten()  # Aplanar la matriz de ejes para acceder más fácilmente

colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'black', 'orange', 'pink']
for i, attribute in enumerate(attributes):
    axs[i].bar(df_graphics.index, df_graphics[attribute], color=colors[i])
    axs[i].set_title(attribute)
    axs[i].set_ylabel(attribute)
    axs[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
st.pyplot(fig)
