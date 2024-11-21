import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58869')
        st.markdown('**Nombre:** Enzo Perez')
        st.markdown('**Comisión:** C9')


def calcular_variaciones(df, productos_orden_original):
    """
    Calcula métricas y variaciones para cada producto.
    """
    resultados = []

    for producto in productos_orden_original:
        if producto in df['producto'].unique():  # Solo procesamos si el producto existe en la sucursal
            grupo = df[df['producto'] == producto].sort_values('año')  # Asegurarse de que se ordene por año
            grupo['precio_promedio'] = grupo['ingreso_total'] / grupo['unidades_vendidas']
            grupo['margen_promedio'] = (grupo['ingreso_total'] - grupo['costo_total']) / grupo['ingreso_total']

            if len(grupo) > 1:
                delta_precio = (grupo['precio_promedio'].iloc[-1] - grupo['precio_promedio'].iloc[-2]) / grupo['precio_promedio'].iloc[-2] * 100
                delta_margen = (grupo['margen_promedio'].iloc[-1] - grupo['margen_promedio'].iloc[-2]) / grupo['margen_promedio'].iloc[-2] * 100
                delta_unidades = (grupo['unidades_vendidas'].iloc[-1] - grupo['unidades_vendidas'].iloc[-2]) / grupo['unidades_vendidas'].iloc[-2] * 100
            else:
                delta_precio = delta_margen = delta_unidades = None

            resultados.append({
                'producto': producto,
                'precio_promedio': grupo['precio_promedio'].mean(),
                'margen_promedio': grupo['margen_promedio'].mean(),
                'unidades_vendidas': grupo['unidades_vendidas'].sum(),
                'delta_precio': delta_precio,
                'delta_margen': delta_margen,
                'delta_unidades': delta_unidades
            })

    return pd.DataFrame(resultados)


def cargar_datos():
    uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = ['sucursal', 'producto', 'año', 'mes', 'unidades_vendidas', 'ingreso_total', 'costo_total']
        df['orden_original'] = df.index  # Guardar el orden original del archivo
        return df, True  # Retornar el DataFrame y un indicador de archivo cargado
    else:
        return None, False  # Retornar None y un indicador de que no hay archivo cargado


def crear_grafico_mejorado(df, producto):
    try:
        # Agrupar datos por año y mes, sumar unidades vendidas e ingresos totales
        ventas = df.groupby(['año', 'mes']).agg({
            'ingreso_total': 'sum',
            'unidades_vendidas': 'sum'
        }).reset_index()

        # Crear la columna de fechas a partir de año y mes
        ventas['fecha'] = pd.to_datetime(ventas['año'].astype(str) + '-' + ventas['mes'].astype(str).str.zfill(2) + '-01')
        ventas = ventas.sort_values('fecha')  # Ordenar por fecha

        # Crear la figura y el eje con un tamaño más grande
        fig, ax = plt.subplots(figsize=(12, 8))  # Aumentar el tamaño de la figura

        # Configurar márgenes en el eje Y
        margen_superior = ventas['unidades_vendidas'].max() * 1.1
        margen_inferior = max(0, ventas['unidades_vendidas'].min() * 0.1)
        ax.set_ylim([margen_inferior, margen_superior])

        # Configurar fondo y cuadrículas
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.grid(True, linestyle='-', alpha=0.3, color='gray')

        # Dibujar línea azul de unidades vendidas
        ax.plot(
            ventas['fecha'], ventas['unidades_vendidas'],
            color='#2563EB', linewidth=1.5, markersize=4, label=producto
        )

        # Calcular y dibujar la línea de tendencia
        z = np.polyfit(np.arange(len(ventas)), ventas['unidades_vendidas'], 1)
        p = np.poly1d(z)
        ax.plot(
            ventas['fecha'], p(np.arange(len(ventas))),
            color='red', linestyle='--', linewidth=1.5, label='Tendencia'
        )

        # Configurar leyenda, títulos y etiquetas
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.set_title('Evolución de Ventas Mensual', fontsize=20, pad=15)
        ax.set_xlabel('Año-Mes', fontsize=10)
        ax.set_ylabel('Unidades Vendidas', fontsize=17)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        # Ajustar el diseño
        plt.tight_layout()

        return fig
    except Exception as e:
        print(f"Error al crear el gráfico para el producto '{producto}': {e}")
        return None


def main():
    # Cargar los datos
    df, archivo_cargado = cargar_datos()

    if not archivo_cargado:
        st.title("Por favor, sube un archivo CSV desde la barra lateral.")
        mostrar_informacion_alumno();
        
        return

    # Configurar selectbox de sucursales
    sucursales = ["Todas"] + list(df['sucursal'].unique())
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

    # Actualizar título dinámicamente
    if sucursal_seleccionada == "Todas":
        st.title("Datos de Todas las Sucursales")
    else:
        st.title(f"Datos de Sucursal {sucursal_seleccionada}")

    if sucursal_seleccionada != "Todas":
        df = df[df['sucursal'] == sucursal_seleccionada]

    # Obtener las variaciones para los productos
    productos_orden_original = df['producto'].unique()
    variaciones = calcular_variaciones(df, productos_orden_original)

    # Mostrar métricas y gráficos
    for _, row in variaciones.iterrows():
        # Usar st.container para crear el contenedor
        with st.container(border= True):
            st.subheader(row['producto'])

            # Dividir la página en columnas
            col1, col2 = st.columns([2, 4])

            with col1:
                # Métricas de la izquierda
                precio_str = f"${row['precio_promedio']:,.0f}".replace(",", ".")
                margen_str = f"{row['margen_promedio']*100:.0f}".replace(".", ",")
                unidades_str = f"{int(row['unidades_vendidas']):,}".replace(",", ".")

                # Mostrar el delta calculado de la variación
                delta_precio = f"{row['delta_precio']:.2f}%" if row['delta_precio'] is not None else "N/A"
                delta_margen = f"{row['delta_margen']:.2f}%" if row['delta_margen'] is not None else "N/A"
                delta_unidades = f"{row['delta_unidades']:.2f}%" if row['delta_unidades'] is not None else "N/A"

                st.metric("Precio Promedio", precio_str, delta=delta_precio)
                st.metric("Margen Promedio", margen_str + "%", delta=delta_margen)
                st.metric("Unidades Vendidas", unidades_str, delta=delta_unidades)

            with col2:
                # Filtrar el DataFrame por el producto actual
                datos_producto = df[df['producto'] == row['producto']]

                # Llamar a la función para crear el gráfico mejorado
                fig = crear_grafico_mejorado(datos_producto, row['producto'])

                if fig:
                    st.pyplot(fig)



if __name__ == "__main__":
    main()
