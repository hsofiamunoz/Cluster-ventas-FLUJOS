# !pip install scikit-learn

import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Cluster ventas")

# ------------------------------------------------------------------
# --------------------- Carga y validacion -------------------------
# ------------------------------------------------------------------
# Función para cargar y procesar el archivo.
def cargar_datos(archivo):
    df = pd.read_excel(archivo)  # Ajusta el nombre de la hoja según tu archivo
    
    # Validar el tipo de archivo de lectura
    columnas_entrada = df.columns

    # validamos el tipo de dato
    if('COLOR' in columnas_entrada):
        df['TIPO_COLOR'] = df.apply(lambda row: f"{row['REFERENCIA']} {row['COLOR']}", axis=1)
        df = df.set_index('TIPO_COLOR') 

    elif('PUNTO_VENTA' in columnas_entrada):
        df = df.set_index('PUNTO_VENTA')
    

    return df

def validar_columnas(df):
    set_columnas = set(df.columns)
    set_obligatorio = set(['CONTRIBUCION', 'ROTACION', 'VENTA_POR_MES', 'MARGEN', 'VENTA_PESOS', 'VENTA_UNDS'])

    if(len(set_obligatorio.intersection(set_columnas)) < 6):
        return False
    else: return True

# ------------------------------------------------------------------
# ------------------- Seleccion de columnas ------------------------
# ------------------------------------------------------------------
def seleccionar_columnas_de_interes(df):
    opciones = df.select_dtypes(include='number').columns.tolist()
    return opciones
    
# ------------------------------------------------------------------
# ------------------------ Modelo  --------------------------------
# ------------------------------------------------------------------
# Función para aplicar PCA y devolver los resultados
def aplicar_pca(df, variables_seleccionadas):

    print(len(variables_seleccionadas))

    # En caso de no seleccionar ninguna variable, se toman las 6 variables por defecto

    if((len(variables_seleccionadas) == 1) & (variables_seleccionadas[0] == '')):
        print(df.select_dtypes(include=['number']).columns)
        variables_explicativas = ['CONTRIBUCION', 'ROTACION', 'VENTA_POR_MES', 'MARGEN', 'VENTA_PESOS', 'VENTA_UNDS']
    else: 
        variables_explicativas = variables_seleccionadas
    
    X = df[variables_explicativas].dropna()
    
    # Estandarización de los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Aplicación de PCA
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    # DataFrame con resultados de PCA
    pca_df = pd.DataFrame(pca_result, columns=[f'z{i+1}' for i in range(pca_result.shape[1])])

    df_result = df.reset_index().merge(pca_df, right_index=True,left_index=True, how='left')
    # ---------------------------------------------------------------------------------------------------
    # Cálculo del índice compuesto
    # Si tenemos mas de 3 variables, entonces tomamos las primeras 3
    len_pca_indices = len(pca.explained_variance_ratio_)

    if(len_pca_indices >=3):
        df_result['Indice'] = df_result['z1'] * pca.explained_variance_ratio_[0] + \
                            df_result['z2'] * pca.explained_variance_ratio_[1] + \
                            df_result['z3'] * pca.explained_variance_ratio_[2]
    elif(len_pca_indices == 2):
        df_result['Indice'] = df_result['z1'] * pca.explained_variance_ratio_[0] + \
                            df_result['z2'] * pca.explained_variance_ratio_[1]
    else:
        df_result['Indice'] = df_result['z1'] * pca.explained_variance_ratio_[0]

    # si queremos tomar todos los factores que influyen:
    # df_result['Indice'] = 0  # Inicializa la columna Indice

    # for i in range(len_pca_indices):
    #     df_result['Indice'] += df_result[f'z{i+1}'] * pca.explained_variance_ratio_[i]

    # original
    # df_result['Indice'] = df_result['z1'] * pca.explained_variance_ratio_[0] + \
    #                     df_result['z2'] * pca.explained_variance_ratio_[1] + \
    #                     df_result['z3'] * pca.explained_variance_ratio_[2]
    

    df_result['Indice'] = df_result['Indice']*(-1)
    # ---------------------------------------------------------------------------------------------------

    # df_result = df_result[['PUNTO_VENTA', 'COD_PUNTO_VENTA', 'CONTRIBUCION', 'ROTACION', 'VENTA_POR_MES', 'MARGEN', 'VENTA_PESOS', 'VENTA_UNDS', 'Indice']]

    df_result = df_result.sort_values(by = 'Indice', ascending=False)
    df_result['ranking_pca'] = range(1, len(df_result) + 1)

    # seleccionamos las columnas que sno tengan z
    columnas_no_z = [col for col in df_result.columns if not col.lower().startswith('z')]

    df_result = df_result[columnas_no_z]

    return df_result


def asignar_grupo_segun_indice(indice):
    if indice >= 1:
        return 'grupo1'
    elif ((indice > 0) & (indice < 1)):
        return 'grupo2'
    elif ((indice < 0) & (indice > -1)):
        return 'grupo3'
    else:
        return 'grupo4'


# ------------------------------------------------------------------
# -------------------------- Graficas  -----------------------------
# ------------------------------------------------------------------
def grafica_resultados(df_result2):
    import plotly.graph_objects as go

    if('COLOR' in df_result2.columns):
        variable_x = 'TIPO_COLOR'
    else:
        variable_x = 'COD_PUNTO_VENTA'

    df_result2 = df_result2.sort_values(by=variable_x, ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Line(
        x = df_result2[variable_x], 
        y = [-1]*len(df_result2[variable_x]),
        line_color='red',
        name='Línea -1'
        ))
                
    fig.add_trace(go.Line(
        x = df_result2[variable_x], 
        y = [1]*len(df_result2[variable_x]),
        line_color='green',
        name='Línea 1'
        ))

    fig.add_trace(go.Line(
        x = df_result2[variable_x], 
        y = [0]*len(df_result2[variable_x]),
        line_color='gray',
        name='Línea 0'
        ))

    fig.add_trace(go.Scatter(
        x = df_result2[variable_x], 
        y = df_result2['Indice'],
        mode='markers',
        marker=dict(color='purple'),
        name='Indice tiendas',
        ))
    return fig

def grafica_pastel(df_result2):
    import plotly.express as px

    grafica_pie  = df_result2.groupby('Grupo')[['Indice']].count().reset_index()
    print(grafica_pie)

    colores = ['green', 'yellowgreen', 'orange', 'red']

    fig = px.pie(grafica_pie, values='Indice', names='Grupo', title='Gráfico de Pie', color='Grupo', 
                 color_discrete_sequence=colores)

    return fig

# ------------------------------------------------------------------
# --------------------- Guardar archivos  --------------------------
# ------------------------------------------------------------------

def to_excel(df: pd.DataFrame):
    from io import BytesIO
    in_memory_fp = BytesIO()
    df.to_excel(in_memory_fp)
    # Write the file out to disk to demonstrate that it worked.
    in_memory_fp.seek(0, 0)
    return in_memory_fp.read()

# ------------------------------------------------------------------
# ----------------------- Menu principal  --------------------------
# ------------------------------------------------------------------
def main():
    df_pca = pd.DataFrame()
    # st.markdown("<h1 style='text-align: center;'>Análisis de Tiendas con PCA</h1>", unsafe_allow_html=True)

    # Página para cargar el archivo
    st.header('Cargar Archivo Excel')
    archivo = st.file_uploader('Cargar archivo Excel', type=['xls', 'xlsx'])

    if archivo is not None:
        # 1. Cargar el archivo
        df = cargar_datos(archivo)

        bandera_validacion = validar_columnas(df)

        # validamos que las 6 variables obligatorias esten bien escritas
        if(bandera_validacion == True):

            st.write('Muestra del archivo cargado:')
            st.write(df.head(5))

            # 2. Seleccion de columnas
            st.header('')
            st.subheader('Seleccionar Columnas de Interés')
            opciones = seleccionar_columnas_de_interes(df)
            seleccion = st.multiselect('Selecciona opciones:', opciones)

            # Mostrar las opciones seleccionadas
            if seleccion:
                st.write(f'Has seleccionado estas columnas: {", ".join(seleccion)}')
                
            else:
                st.write('Aún no has seleccionado ninguna columna.')

            # # Página para mostrar resultados después de aplicar PCA
            st.header('')
            if st.button('Generar Resultados'):

                columnas_seleccionadas = ','.join(seleccion).split(',')
                print(columnas_seleccionadas)
                
                df_pca = aplicar_pca(df, variables_seleccionadas=columnas_seleccionadas)

                df_pca['Grupo'] = df_pca['Indice'].apply(lambda x: asignar_grupo_segun_indice(x))
                
                st.write('Muestra de los resultados:')
                st.write(df_pca.head(5))

                st.write('Grafica de resultados')
                fig = grafica_resultados(df_pca)
                st.plotly_chart(fig)

                fig2 = grafica_pastel(df_pca)
                st.plotly_chart(fig2)

            # Boton de descarga del archivo
            excel_data = to_excel(df_pca)
            file_name = "resultados_modelo_pca.xlsx"
            st.download_button(
                f" Descargar resultados",
                excel_data,
                file_name,
                f"text/{file_name}",
                key=file_name
            )

        else:
            st.header('')
            st.subheader('Debes revisar el formato de las columnas de entrada')



if __name__ == '__main__':
    main()
