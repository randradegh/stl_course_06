
##
# Análisis de campañas de marketing.
##
# Fecha inicio: 20220125
# Fuente de datos: Fuente: https://www.mavenanalytics.io/data-playground?page=2

##
# Utilerías del proyecto
##
# Debe ser el primer comando

from curses import echo
from utils import *

##
#
## Librerías adicionales para esta app
import pydeck as pdk
from pydeck.types import String
from sklearn.linear_model import LinearRegression

#st.image('images/wp3205208.jpg')
#st.markdown(my_footer, unsafe_allow_html=True)
header("9 casos de negocio con Streamlit")
st.subheader("6. Análisis de Campañas de Marketing.")
st.subheader("Introducción")

st.markdown("""
    El objetivo de esta app es revisar la efectividad de algunas campañas de marketing usando varias técnicas.

    El _dataset_ de esta sesión contiene diversos _features_ de algunas campañas de 
    marketing: datos del comprador, el monto de sus compras de artículos de varias 
    categorías, datos sobre sus hijos, sobre la tienda en donde hizo la compra, 
    país en donde reside, cantidad de visitas al sitio web, etc.

    Vamos a utilizar esa información para poder determinar que campañas de marketing fueron lás más exitosas.

    ___

    ## Temas a analizar
    **Preguntas que dirigen el análisis:**
    - ¿Hay valores nulos o atípicos? ¿Cómo contender con ellos?
    - ¿Cómo es el cliente promedio?
    - ¿Qué productos funcionan mejor?
    - ¿Qué campaña de marketing fue la más exitosa? _**Será resuelta por los alumnos**_.
    - ¿Qué canales tienen un rendimiento inferior? _**Será resuelta por los alumnos**_.
    - ¿Qué factores están significativamente relacionados con el número de compras por medio del catálogo?

    ### Carga de datos y contenido del _dataset_
    Cargamos los datos y mostramos los primeros cinco registros:
""")

with st.echo(code_location='bellow'):

    df_mkt = pd.read_csv("mkt_data/marketing_data_cleaned.csv", sep=',')
    st.write(df_mkt.head(5))



'''
### Diccionario  de datos (*Data Dictionary*)

**Customer Profile**

- ID: Customer's unique identifier
- Year_Birth: Customer's birth year
- Education: Customer's education level
- Marital_Status: Customer's marital status
- Income: Customer's yearly household income
- Kidhome: Number of children in customer's household
- Teenhome: Number of teenagers in customer's household
- Dt_Customer: Date of customer's enrollment with the company
- Recency: Number of days since customer's last purchase
- Complain: 1 if customer complained in the last 2 years, 0 otherwise
- Country: Customer's location

**Product Preferences**

- MntWines: Amount spent on wine in the last 2 years
- MntFruits: Amount spent on fruits in the last 2 years
- MntMeatProducts: Amount spent on meat in the last 2 years
- MntFishProducts: Amount spent on fish in the last 2 years
- MntSweetProducts: Amount spent on sweets in the last 2 years
- MntGoldProds: Amount spent on gold in the last 2 years

*Channel Performance*

- NumWebPurchases: Number of purchases made through the company's web site
- NumCatalogPurchases: Number of purchases made using a catalogue
- NumStorePurchases: Number of purchases made directly in stores
- NumWebVisitsMonth: Number of visits to company's web site in the last month

*Campaign Success*

- NumDealsPurchases: Number of purchases made with a discount
- AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
- AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
- AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
- AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
- AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
- Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
'''

'''
    ___

    ### Análisis inicial con el apoyo del método _describe()_
'''

st.markdown("""
Haremos un somero análisis estadístico del *dataset* con la función _describe()_ de Python. 

La función _describe()_ se utiliza para calcular algunos datos estadísticos como percentiles, media y desviacion 
estándar de los valores numéricos de nuestros datos.
""")
# calling describe method
desc = df_mkt.describe()
# display
st.dataframe(desc)

#st.warning("Poner atención a los valores máximo, medio y promedio.")
st.info("Poner atención a los valores máximo, medio y promedio.")

##
# Fuente: # https://www.askpython.com/python/examples/detection-removal-outliers-in-python
#
'''
    ## Valores atípicos (_Outliers_)

    Un valor atípico es un punto o conjunto de puntos de datos que se 
    encuentran alejados del resto de los valores de datos del conjunto de 
    datos. Es decir, son puntos de datos que aparecen lejos de la distribución 
    general de valores de datos en un conjunto de datos.

    Básicamente, los valores atípicos parecen divergir de la distribución 
    general adecuada y bien estructurada de los elementos de datos. 
    Puede considerarse como una distribución anormal que aparece fuera 
    de la clase o población.

    ¿Por qué es necesario eliminar los valores atípicos de los datos?
    Como se discutió anteriormente, los valores atípicos son los datos 
    que se encuentran fuera de la distribución habitual de los datos 
    y causan los siguientes efectos en la distribución general de datos:

    - Afecta a la variación estándar general de los datos.
    - Manipula la media general de los datos.
    - Convierte los datos a una forma sesgada.
    - Provoca un sesgo en la estimación de la precisión del modelo de aprendizaje automático.
    - Afecta la distribución y las estadísticas del conjunto de datos.
    ___
    **Detección de valores atípicos: enfoque IQR**

    Los valores atípicos en el conjunto de datos se pueden detectar mediante 
    los siguientes métodos:

    - Z-score
    - Gráfico de dispersión
    - Rango intercuartil (_Interquertil Range, IQR_)

    Implementaremos el método IQR para detectar y tratar valores atípicos.

    IQR es el acrónimo de *Interquartile Range*. Mide la dispersión estadística 
    de los valores de los datos como una medida de la distribución general.

    IQR es equivalente a la diferencia entre el primer cuartil (Q1) y el 
    tercer cuartil (Q3) respectivamente.

    Aquí, Q1 se refiere al primer cuartil, es decir, 25 %, y Q3 se refiere 
    al tercer cuartil, es decir, 75 %.

    Usaremos Boxplots para detectar y visualizar los valores atípicos 
    presentes en el conjunto de datos.

    Los diagramas de caja representan la distribución de los datos en términos 
    de cuartiles y consta de los siguientes componentes:

    Observemos gráficamente que son los siguientes conceptos:
    - Q1-25%
    - Q2-50%
    - Q3-75%
    - Límite inferior
    - Límite superior
'''
st.image('images/Detection-of-Outlier-BoxPlot-1.png', caption='Boxplot', width=500)


st.markdown("## En busca de los _outliers_")
st.markdown("### _Box Plot_ de los ingresos de los clientes.")
st.markdown("#### ¿Que és un _Box Plot_?")
st.markdown("""
    En estadística descriptiva, un diagrama de caja (también conocido como diagrama de caja y bigotes) es un tipo de gráfico que se 
    utiliza a menudo en el análisis exploratorio de datos. Los diagramas de caja muestran visualmente la distribución de datos numéricos 
    y la asimetría mediante la visualización de los cuartiles de datos (o percentiles) y algún valor de tendencia central.
""")

with st.echo(code_location='above'):
    # Box_plot para income
    #df_income = df_mkt(['income'])
    BGCOLOR = "#0431B4"

    fig_bp = px.box(df_mkt, y="Income", title='Box Plot para el ingreso de los clientes')

    fig_bp.update_layout(width=600,height=500)
    fig_bp.update_yaxes(title_text='Ingreso')
    fig_bp.update_xaxes(title_text='')

    fig_bp.update_layout({
            'paper_bgcolor': BGCOLOR,
            'width' : 700,
            'height': 500,
            'title': 'Boxplot de los ingresos de los clientes.'
        })

    fig_bp

'''
    #### Análisis de los montos de los gastos de los clientes, por tipo de artículo.
    Para analizar los _features_ referidos a los montos de los gastos de los clientes 
    en vinos, fruta, carne, pescado, golosinas y oro vamos a visualizar sus _boxplots_.

    Podremos ver los detalles de cada uno de ellos gracias a la posibilidad de 
    hacer un acercamiento (_zoom_) a cualquiera de ellos.
'''

with st.echo(code_location='above'):
    BGCOLOR = "#0431B4"
    import plotly.graph_objects as go

    df_mnt = df_mkt[['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']]
    fig_mnt = go.Figure()
    for col in df_mnt:
        fig_mnt.add_trace(go.Box(y=df_mnt[col].values, name=df_mnt[col].name))

    fig_mnt.update_layout({
            'paper_bgcolor': BGCOLOR,
            'width' : 900,
            'height': 600,
            'title': 'Análisis de valores atípicos (outliers) en algunas líneas de productos'
        })
    fig_mnt

'''
    **Eliminación de valores atípicos**

    Ahora es el momento de tratar los valores atípicos que hemos detectado usando Boxplot en la sección anterior.

    Usando IQR, podemos seguir el siguiente enfoque para reemplazar los valores atípicos con un valor NULL:

    - Calcular el primer y tercer cuartil (Q1 y Q3).
    - Además, evalúe el rango intercuartílico, IQR = Q3-Q1.
    - Estime el límite inferior, el límite inferior = Q1*1.5
    - Estime el límite superior, límite superior = Q3*1.5
    - Reemplace los puntos de datos que se encuentran fuera del límite inferior y superior con un valor NULL.

'''
with st.echo(code_location='above'):
    for x in df_mnt:
        q75,q25 = np.percentile(df_mnt.loc[:,x],[75,25])
        intr_qr = q75-q25
    
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
    
        df_mnt.loc[df_mnt[x] < min,x] = np.nan
        df_mnt.loc[df_mnt[x] > max,x] = np.nan


st.write(df_mnt.describe())

with st.echo(code_location='above'):
    BGCOLOR = "#0431B4"
    import plotly.graph_objects as go

    fig_mnt2 = go.Figure()
    for col in df_mnt:
        fig_mnt2.add_trace(go.Box(y=df_mnt[col].values, name=df_mnt[col].name))

    fig_mnt2.update_layout({
            'paper_bgcolor': '#0B0B3B',
            'width' : 700,
            'height': 500,
            'title': 'Análisis en algunas líneas de productos, SIN outliers'
        })

with st.echo(code_location='bellow'):
    col_1, col_2 = st.columns(2)
    with col_1:
        fig_mnt.update_layout({
            'width' : 700,
            'height': 500,
            'title': 'Análisis de algunas líneas de productos, CON outliers'
        })
        fig_mnt
    with col_2:
        fig_mnt2


'''
___
### Análisis del cliente promedio

Elegimos los _features_ que definan mejor al cliente y obtenemos su valor promedio.
'''
with st.echo(code_location='bellow'):
    # Creamos lo dos dataframes que utilizaremos, el primero para la narrativa
    df_mean_client = df_mkt[['Year_Birth','Income','Kidhome','Teenhome','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth']].copy().mean()

    """Media de Datos de Clientes"""
    st.write(df_mean_client)
    # y el segundo para las cantidades
    df_mean_client_num = df_mkt[['NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth']].copy().mean()

    """Media de Datos de Comprasxzzzz"""
    st.write(df_mean_client_num)
    
    df_mean_client_num = df_mean_client_num.to_frame()
    df_mean_client_num = df_mean_client_num.reset_index(drop=False)

    df_mean_client_num.rename(columns= {'index':'Tipo', 0:'Cantidad'}, inplace=True)

    # Obtenemos el año actual
    import datetime

    currentDateTime = datetime.datetime.now()
    date = currentDateTime.date()
    year = int(date.strftime("%Y"))
    #st.write(year)

    '''
        #### Con el _dataset_ obtenido podemos hacer una narrativa del cliente:
    '''
    cola, colb = st.columns(2)
    with cola:
        st.write(f"""
            #### Nuestro Cliente Promedio
            - Nuestro cliente promedio nació en {int(df_mean_client['Year_Birth'])}, es decir tiene {year - int(df_mean_client['Year_Birth'])} años.

            - Sus ingresos son de {round(df_mean_client['Income'])} USD, tiene {round(df_mean_client['Kidhome'],2)} hijos pequeños y {round(df_mean_client['Teenhome'],2)} hijos adolescentes.

            - Sus cantidades promedio de compras por tipo son:
                - Compras con descuento: {round(df_mean_client['NumDealsPurchases'],2)}
                - Compras hechas en la web de la tienda: {round(df_mean_client['NumWebPurchases'],2)}
                - Compras hechas usando un catálogo: {round(df_mean_client['NumCatalogPurchases'],2)}
                - Compras directamente en tiendas: {round(df_mean_client['NumStorePurchases'],2)}
                - Cantidad de visitas al sitio web de la tienda en el último mes: {round(df_mean_client['NumWebVisitsMonth'],2)}
            """)

    
    with colb:
        color = '#1A5276'
        
        fig1 = px.bar(df_mean_client_num, y='Tipo', x='Cantidad', color='Tipo', orientation='h', 
            #template='presentation', 
            title="Promedio de Compras y Visitas a la Web"
            )
        
        fig1.update_yaxes(title_text='Tipo de Compra o Visita')
        fig1.update_xaxes(title_text='Cantidad de Compras o Visitas')
        
        fig1.update_layout({
                'font_color' : '#2C3E50',
                'plot_bgcolor': color,
                'paper_bgcolor': color,
                'width': 700,
                'height': 400
        })

        fig1.update_yaxes(visible=False, showticklabels=False)

        # Mostramos el gráfico
        fig1



'''
    ___
'''
##
# Análisis sobre los productos que funcionan mejor
##    
'''
    ## ¿Qué productos funcionan mejor?
'''
df_products = df_mkt[['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']]
df_products_sum = df_products.sum()
df_mean_prod = df_products_sum.to_frame()
df_products_sum = df_products_sum.reset_index(drop=False)
df_products_sum.rename(columns= {'index':'Tipo', 0:'Suma'}, inplace=True)

#st.write(df_products_sum['Suma'][df_products_sum['Tipo']=='MntFruits'])
wines = int(df_products_sum['Suma'][df_products_sum['Tipo']=='MntWines'].iloc[0])
fruits = int(df_products_sum['Suma'][df_products_sum['Tipo']=='MntFruits'].iloc[0])
meat = int(df_products_sum['Suma'][df_products_sum['Tipo']=='MntMeatProducts'].iloc[0])
fish = int(df_products_sum['Suma'][df_products_sum['Tipo']=='MntFishProducts'].iloc[0])
sweet = int(df_products_sum['Suma'][df_products_sum['Tipo']=='MntSweetProducts'].iloc[0])
gold = int(df_products_sum['Suma'][df_products_sum['Tipo']=='MntGoldProds'].iloc[0])


df_mean_prod = df_products.mean()
df_mean_prod = df_mean_prod.to_frame()
df_mean_prod = df_mean_prod.reset_index(drop=False)
df_mean_prod.rename(columns= {'index':'Tipo', 0:'Promedio'}, inplace=True)

# Ocultamos el despliegue de los df
with st.expander("Visualizar/Ocultar los dataframes de esta sección", expanded=False):
    '''
        **Suma de todos los productos**
    '''
    st.write(df_products_sum)
    '''
        **Promedio de todos los productos**
    '''
    st.write(df_mean_prod)

'''
    #### Cantidad total de productos vendidos
'''

fig = go.Figure()
font_color_text = '#7FB3D5'
font_color_number = '#82E0AA'
font_size = 40
col01, col02, col03, col04, col05 = st.columns(5)

with col01:
    ref = 1
    fig.add_trace(go.Indicator(
        mode = "number",
        number = {'font.size' : font_size, 'font.color': font_color_number, 'valueformat':','},
        value = wines,
        title = {'text': 'Vinos', 'font.size': 25, 'font.color':font_color_text},
        domain = {'row': 0, 'column': 0}))

with col02:
    ref = 1
    fig.add_trace(go.Indicator(
        mode = "number",
        number = {'font.size' : font_size, 'font.color': font_color_number, 'valueformat':','},
        value = fruits, 
        title = {'text': 'Frutas', 'font.size': 25, 'font.color':font_color_text},
        domain = {'row': 0, 'column': 1}))

with col03:
    ref = 1
    fig.add_trace(go.Indicator(
        number = {'font.size' : font_size, 'font.color': font_color_number, 'valueformat':','},
        value = meat,
        title = {'text': 'Carnes', 'font.size': 25, 'font.color':font_color_text},
        domain = {'row': 0, 'column': 2}))
#format(num, ",")
with col04:
    ref = 1
    fig.add_trace(go.Indicator(
        mode = "number",
        number = {'font.size' : font_size, 'font.color': font_color_number, 'valueformat':','},
        value = fish,
        title = {'text': 'Pescados', 'font.size': 25, 'font.color':font_color_text},
        domain = {'row': 0, 'column': 3}))

with col05:
    ref = 1
    fig.add_trace(go.Indicator(
        mode = "number",
        number = {'font.size' : font_size, 'font.color': font_color_number, 'valueformat':','},
        value = gold,
        title = {'text': 'Oro', 'font.size': 25, 'font.color':font_color_text},
        domain = {'row': 0, 'column': 4}))

fig.update_layout(
    width = 1000, 
    height = 200,
    paper_bgcolor = "#042f47", 
    margin=dict(l=20, r=20, t=20, b=5),
    grid = {'rows': 1, 'columns': 5, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'mode' : "number"}]
        }}
)
fig

st.write('**¿Qué gráfico es mejor para conocer la situación de los promedios**')
fig_mean = go.Figure()

col_k1, col_k2 = st.columns(2)
#col01, col02, col03, col04, col05 = st.columns(5)

with st.container():

    with col_k1:
        color = '#154360'
        fig_mean = px.bar(df_mean_prod, y='Tipo', x='Promedio', color='Tipo', orientation='h', 
            #template='gridon', 
            title="Promedio de Artículos Vendidos"
            )

        fig_mean.update_layout(width=900,height=500)
        fig_mean.update_yaxes(title_text='Tipo de Artículo')
        fig_mean.update_xaxes(title_text='Promedio')

        fig_mean.update_layout({
                'font_color' : '#2C3E50',
                'plot_bgcolor': color,
                'paper_bgcolor': color,
                'width': 600,
                #'height': 400
        })

        # Mostramos el gráfico
        fig_mean

    with col_k2:
        #st.success("From Col1")
        color = '#154360'
        fig_mean = px.pie(df_mean_prod, names='Tipo', values='Promedio', color='Tipo', 
            template='gridon', 
            title="Promedio de Artículos Vendidos",
            )

        fig_mean.update_layout(width=900,height=500)
        fig_mean.update_yaxes(title_text='Tipo de Artículo')
        fig_mean.update_xaxes(title_text='Promedio')

        fig_mean.update_layout({
                'font_color' : '#2C3E50',
                'plot_bgcolor': color,
                'paper_bgcolor': color,
                'width': 600,
                #'height': 400,
                #'margin': dict(l=0, r=0, t=0, b=0),
        })
        
        # Mostramos el gráfico
        fig_mean

##
# ¿Que feature predice mejor las compras?
##

'''
    ___
    ### Análisis sobre que rasgos, características o _features_ predices mejor las compras
'''

'''
    #### **Importancia de los _features_**

    La importancia de los _features_ radica en el conocimiento de qué campos tuvieron el mayor impacto en cada 
    predicción generada por clasificación o análisis de regresión. Cada valor de los _features_ tiene una 
    magnitud y una dirección (positiva o negativa), que indican cómo cada 
    campo (o característica de un punto de datos) afecta una predicción particular.

    El propósito de la clasificación de los _features_ es ayudarnos a determinar si las predicciones 
    son sensatas. ¿La relación entre la variable dependiente y las características importantes está respaldada 
    por su conocimiento del dominio?
    
    Las lecciones que aprendamos  sobre la importancia de características específicas también 
    pueden afectar nuestra decisión de incluirlas en iteraciones futuras de algún modelo entrenado.

    La base muchas de las técnicas de aprendizaje automático (_machine learning, ML_) son las 
    regresiones lineales.

    Haremos uso de ese tipo de regresión para tratar de descubrir que _feature_ o rasgo nos permite 
    predecir la cantidad de ventas por catálogo.

    ##### Matriz de Regresiones Lineales para algunos rasgos seleccionados
'''

# Correlación de features
df_corr = df_mkt[['NumWebPurchases','NumDealsPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','Response','Complain','Country']]


fig_corr  = px.scatter_matrix(df_mkt,
    dimensions=['NumWebPurchases','NumDealsPurchases', 'Income','NumCatalogPurchases', 'NumStorePurchases']
    #dimensions=['Income','NumCatalogPurchases']
    )


fig_corr.update_layout({
        'paper_bgcolor': '#0B0B3B',
        'width' : 1000,
        'height': 1000,
        'title': 'Correlación entre features seleccionados de la fuente de datos.'
    })

fig_corr.update_traces(diagonal_visible=False)

fig_corr



'''
    ### Regresión lineal o ¿el nivel de ingresos en un buen _feature_ para las compras por catálago?
'''

##
# Nota sobre feature selection
# Fuente: https://www.datacamp.com/community/tutorials/feature-selection-python
# Understanding the importance of feature selection
# The importance of feature selection can best be recognized when you 
# are dealing with a dataset that contains a vast number of features. 
# This type of dataset is often referred to as a high dimensional 
# dataset. Now, with this high dimensionality, comes a lot of problems 
# such as - this high dimensionality will significantly increase the training 
# time of your machine learning model, it can make your model very 
# complicated which in turn may lead to Overfitting.

'''
    Para investigar si la variable de ingreso predice la cantidad de compras por catálogo 
    usaremos un modelo de regresión líneal usando _sklearn_.

    Creamos un df para la regresión y eliminamos los NAN pues el modelo lo requiere.

    A continuación usamos el método _reshape()_ que crea una matriz de una columna, manteniendo la congruencia de x y y.
    
    Luego convertimos la columna del dataframe en un arreglo de NumPy.
'''

with st.echo(code_location='above'):
    
    df_lr = df_mkt[['Income', 'NumCatalogPurchases']].dropna()
    
    # Matrices para la regresión lineal
    x = df_lr['Income'].to_numpy().reshape((-1,1))
    y = df_lr['NumCatalogPurchases'].to_numpy()

'''
    Creamos el modelo de regresión lineal con los arreglos generados.
'''

with st.echo(code_location='above'):
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    st.write('coefficient of determination:', r_sq)
    st.write("Intercept:", model.intercept_)
    st.write("Pendiente:", "%.4f" % model.coef_[0])

'''
    Ecuación de la recta usando los valores del modelo
    mnt_cat = (0.0001 * income) - 2.16179
'''
with st.echo(code_location='above'):
    Fuente = 'Modelo'
    df_inc_mnt = []
    for i in range(1700, 115800, 10000):
        mnt_cat = (model.coef_[0] * i) + model.intercept_
        df_inc_mnt.append((i, mnt_cat,Fuente))
        
    df_inc_mnt = pd.DataFrame(df_inc_mnt, columns=('Income', 'NumCatalogPurchases','Fuente'))

    # Añadimos la columna de color a df_mkt
    df_mkt['Fuente'] = 'Dataset'
    frames = [df_mkt[['Income','NumCatalogPurchases','Fuente']], df_inc_mnt]
    df_both = pd.concat(frames)
    #st.write(df_both)
    fig_reg = px.scatter(df_both, x='Income',y='NumCatalogPurchases',color='Fuente')


fig_reg.update_layout(
    title="Pruebas", 
    xaxis_title="Ingreso",
    yaxis_title="Cantidada de Compras por Catálogo",
        font=dict(
        size= 15
    )
)
BGCOLOR = '#0A122A'
fig_reg.update_layout({
        'plot_bgcolor': BGCOLOR,
        'paper_bgcolor': BGCOLOR,
        'template': 'xgridoff',
        'width' : 800,
        'height': 600,
        'title': 'Datos del dataset vs datos del modelo de regresión lineal.'
    })
fig_reg

