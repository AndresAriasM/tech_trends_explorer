# src/database_manager.py - SOLO DYNAMODB
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from data_storage import initialize_database

def run_database_manager(db=None):
    """
    Ejecuta la interfaz de gestión de la base de datos DynamoDB.
    Permite ver, organizar y analizar los datos guardados.
    """
    st.title("📊 Gestor de Datos DynamoDB")
    
    # Si no se proporciona db, intentar inicializar con DynamoDB
    if db is None:
        try:
            # Verificar credenciales AWS
            aws_configured = (
                st.session_state.get('aws_access_key_id') and 
                st.session_state.get('aws_secret_access_key') and 
                st.session_state.get('aws_region')
            )
            
            if aws_configured:
                db = initialize_database(
                    "dynamodb",
                    region_name=st.session_state.aws_region,
                    aws_access_key_id=st.session_state.aws_access_key_id,
                    aws_secret_access_key=st.session_state.aws_secret_access_key
                )
            else:
                st.error("❌ Se requieren credenciales de AWS para acceder a DynamoDB")
                st.info("Configura las credenciales en el panel lateral")
                return
        except Exception as e:
            st.error(f"Error al inicializar DynamoDB: {str(e)}")
            return
    
    if db is None:
        st.error("No se pudo inicializar el sistema de almacenamiento DynamoDB.")
        return
    
    # Mostrar información de conexión
    st.success("🗄️ **Conectado a DynamoDB**")
    st.info(f"📍 Región: {st.session_state.get('aws_region', 'No especificada')}")
    
    # Opciones de la interfaz
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Ver Análisis Guardados", 
        "📂 Gestionar Categorías", 
        "📈 Análisis Comparativo",
        "⚙️ Configuración AWS"
    ])
    
    with tab1:
        show_saved_analyses(db)
    
    with tab2:
        manage_categories(db)
    
    with tab3:
        comparative_analysis(db)
    
    with tab4:
        show_aws_configuration()

def show_aws_configuration():
    """Muestra la configuración de AWS"""
    st.header("⚙️ Configuración de AWS DynamoDB")
    
    st.write("""
    ### Configuración de Credenciales
    
    Para usar DynamoDB, necesitas configurar las credenciales de AWS:
    """)
    
    # Mostrar estado actual
    aws_configured = (
        st.session_state.get('aws_access_key_id') and 
        st.session_state.get('aws_secret_access_key') and 
        st.session_state.get('aws_region')
    )
    
    if aws_configured:
        st.success("✅ Credenciales de AWS configuradas")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Región:** {st.session_state.get('aws_region')}")
        with col2:
            masked_key = st.session_state.get('aws_access_key_id', '')[:8] + "..."
            st.info(f"**Access Key:** {masked_key}")
    else:
        st.warning("⚠️ Credenciales de AWS no configuradas")
    
    st.write("""
    ### Tablas DynamoDB Requeridas
    
    El sistema requiere las siguientes tablas:
    
    1. **tech-trends-analyses**
       - Partition Key: `analysis_id` (String)
       - Sort Key: `timestamp` (String)
    
    2. **tech-trends-categories**
       - Partition Key: `category_id` (String)
    
    ### Configurar Credenciales
    
    Puedes configurar las credenciales de AWS de varias formas:
    
    1. **En el panel lateral de la aplicación**
    2. **Variables de entorno:**
       ```bash
       export AWS_ACCESS_KEY_ID=tu_access_key
       export AWS_SECRET_ACCESS_KEY=tu_secret_key
       export AWS_DEFAULT_REGION=us-east-1
       ```
    
    3. **Archivo de credenciales AWS (~/.aws/credentials)**
    """)
    
    # Probar conexión
    if st.button("🔗 Probar Conexión DynamoDB"):
        if aws_configured:
            try:
                db = initialize_database(
                    "dynamodb",
                    region_name=st.session_state.aws_region,
                    aws_access_key_id=st.session_state.aws_access_key_id,
                    aws_secret_access_key=st.session_state.aws_secret_access_key
                )
                if db:
                    st.success("✅ Conexión exitosa a DynamoDB")
                    
                    # Mostrar información de las tablas
                    try:
                        categories = db.get_all_categories()
                        analyses = db.storage.get_all_searches()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Categorías", len(categories))
                        with col2:
                            st.metric("Análisis", len(analyses))
                    except Exception as e:
                        st.warning(f"Conexión exitosa, pero error al leer datos: {str(e)}")
                else:
                    st.error("❌ No se pudo establecer conexión")
            except Exception as e:
                st.error(f"❌ Error de conexión: {str(e)}")
        else:
            st.error("❌ Configura las credenciales primero")

def show_saved_analyses(db):
    """Muestra todos los análisis guardados con opciones de filtrado."""
    st.header("🔍 Análisis Guardados en DynamoDB")
    
    # Obtener categorías y análisis
    try:
        categories = db.get_all_categories()
        all_analyses = db.storage.get_all_searches()
    except Exception as e:
        st.error(f"Error al cargar datos de DynamoDB: {str(e)}")
        return
    
    if not all_analyses:
        st.info("No hay análisis guardados en DynamoDB. Realiza algunas búsquedas en las pestañas de análisis.")
        return
    
    # Opciones de filtrado
    st.subheader("Filtrar Análisis")
    
    # Por categoría
    category_options = {"Todas las categorías": "all"}
    category_options.update({cat["name"]: cat["category_id"] for cat in categories})
    
    selected_category = st.selectbox(
        "Categoría",
        options=list(category_options.keys()),
        index=0,
        key="db_manager_category_filter_selectbox_dynamodb"
    )
    selected_category_id = category_options[selected_category]
    
    # Filtrar por fecha
    date_range = st.date_input(
        "Rango de fechas",
        value=[
            datetime.now().replace(year=datetime.now().year-1).date(),
            datetime.now().date()
        ],
        help="Filtra análisis por fecha de ejecución",
        key="db_manager_date_range_input_dynamodb"
    )
    
    # Aplicar filtros
    filtered_analyses = []
    
    for analysis in all_analyses:
        # Filtrar por categoría
        if selected_category_id != "all" and analysis.get("category_id") != selected_category_id:
            continue
            
        # Filtrar por fecha
        try:
            timestamp = analysis.get("timestamp") or analysis.get("execution_date")
            if timestamp:
                analysis_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                if len(date_range) == 2:
                    if not (date_range[0] <= analysis_date <= date_range[1]):
                        continue
        except Exception:
            pass
        
        filtered_analyses.append(analysis)
    
    # Mostrar resultados
    st.subheader(f"Resultados ({len(filtered_analyses)} análisis)")
    
    if not filtered_analyses:
        st.info("No hay análisis que coincidan con los filtros seleccionados.")
        return
    
    # Crear tabla de resultados
    analyses_data = []
    
    for analysis in filtered_analyses:
        # Obtener nombre de categoría
        category_name = "Desconocida"
        for cat in categories:
            if cat.get("category_id") == analysis.get("category_id"):
                category_name = cat.get("name", "Sin nombre")
                break
        
        analysis_id = analysis.get("analysis_id") or analysis.get("id")
        
        timestamp = analysis.get("timestamp") or analysis.get("execution_date")
        try:
            if timestamp:
                formatted_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime("%d/%m/%Y %H:%M")
            else:
                formatted_date = "Fecha desconocida"
        except Exception:
            formatted_date = str(timestamp) if timestamp else "Fecha desconocida"
        
        analyses_data.append({
            "ID": analysis_id or "Sin ID",
            "Nombre": analysis.get("name") or "Sin nombre",
            "Categoría": category_name,
            "Fecha": formatted_date,
            "Consulta": analysis.get("query") or analysis.get("search_query") or "Sin consulta",
            "Tipo": analysis.get("analysis_type") or "Desconocido",
            "Datos de Papers": "Sí" if analysis.get("paper_data") else "No",
            "Datos de Patentes": "Sí" if analysis.get("patent_data") else "No"
        })
    
    # Mostrar tabla
    df_analyses = pd.DataFrame(analyses_data)
    st.dataframe(
        df_analyses,
        use_container_width=True,
        column_config={
            "ID": st.column_config.TextColumn("ID", width="small"),
            "Nombre": st.column_config.TextColumn("Nombre", width="medium"),
            "Categoría": st.column_config.TextColumn("Categoría", width="small"),
            "Fecha": st.column_config.TextColumn("Fecha", width="small"),
            "Consulta": st.column_config.TextColumn("Consulta", width="large"),
            "Tipo": st.column_config.TextColumn("Tipo", width="small"),
            "Datos de Papers": st.column_config.TextColumn("Papers", width="small"),
            "Datos de Patentes": st.column_config.TextColumn("Patentes", width="small")
        }
    )
    
    # Ver detalles de un análisis específico
    st.subheader("Ver Detalles de Análisis")
    
    if filtered_analyses:
        analysis_options = {}
        for analysis in filtered_analyses:
            analysis_id = analysis.get("analysis_id") or analysis.get("id")
            analysis_name = analysis.get("name") or f"Análisis {analysis_id}"
            analysis_options[analysis_name] = analysis_id
        
        selected_analysis_name = st.selectbox(
            "Selecciona un análisis para ver detalles",
            options=list(analysis_options.keys()),
            key="db_manager_analysis_details_selectbox_dynamodb"
        )
        
        selected_analysis_id = analysis_options[selected_analysis_name]
        
        if selected_analysis_id:
            display_analysis_details(db, selected_analysis_id)

def display_analysis_details(db, analysis_id):
    """Muestra los detalles de un análisis específico."""
    try:
        analysis = db.get_analysis_by_id(analysis_id)
    except Exception as e:
        st.error(f"Error al obtener análisis: {str(e)}")
        return
    
    if not analysis:
        st.error(f"No se encontró el análisis con ID: {analysis_id}")
        return
    
    analysis_name = analysis.get('name') or 'Análisis sin nombre'
    with st.expander(f"Detalles: {analysis_name}", expanded=True):
        # Información básica
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Consulta:**", analysis.get("query") or analysis.get("search_query", "No especificada"))
            
            timestamp = analysis.get("timestamp") or analysis.get("execution_date")
            if timestamp:
                try:
                    formatted_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime("%d/%m/%Y %H:%M")
                except Exception:
                    formatted_date = str(timestamp)
            else:
                formatted_date = "Fecha desconocida"
            
            st.write("**Fecha de ejecución:**", formatted_date)
        
        with col2:
            try:
                category = db.storage.get_category_by_id(analysis.get("category_id"))
                category_name = category.get("name") if category else "Desconocida"
            except Exception:
                category_name = "Error al obtener categoría"
            
            st.write("**Categoría:**", category_name)
            st.write("**Tipo de análisis:**", analysis.get("analysis_type", "No especificado"))
            
            paper_count = len(analysis.get("paper_data", {}))
            patent_count = len(analysis.get("patent_data", {}))
            st.write(f"**Datos:** {paper_count} años de papers, {patent_count} años de patentes")
        
        # Mostrar datos en tablas si existen
        if analysis.get("paper_data"):
            st.subheader("Datos de Papers")
            
            paper_data = analysis["paper_data"]
            df_papers = pd.DataFrame({
                "Año": list(paper_data.keys()),
                "Cantidad": list(paper_data.values())
            })
            df_papers["Acumulado"] = df_papers["Cantidad"].cumsum()
            
            st.dataframe(df_papers, use_container_width=True)
            
            if analysis.get("paper_metrics"):
                metrics = analysis["paper_metrics"]
                
                cols = st.columns(3)
                with cols[0]:
                    r2_value = metrics.get('R2', metrics.get('r2', 0))
                    st.metric("R² del ajuste", f"{r2_value:.4f}")
                with cols[1]:
                    x0_value = metrics.get('x0', metrics.get('punto_inflexion', 0))
                    st.metric("Punto de inflexión", f"{x0_value:.1f}")
                with cols[2]:
                    fase = metrics.get("Fase", metrics.get("fase", "No disponible"))
                    st.metric("Fase", fase)
        
        if analysis.get("patent_data"):
            st.subheader("Datos de Patentes")
            
            patent_data = analysis["patent_data"]
            df_patents = pd.DataFrame({
                "Año": list(patent_data.keys()),
                "Cantidad": list(patent_data.values())
            })
            df_patents["Acumulado"] = df_patents["Cantidad"].cumsum()
            
            st.dataframe(df_patents, use_container_width=True)
            
            if analysis.get("patent_metrics"):
                metrics = analysis["patent_metrics"]
                
                cols = st.columns(3)
                with cols[0]:
                    r2_value = metrics.get('R2', metrics.get('r2', 0))
                    st.metric("R² del ajuste", f"{r2_value:.4f}")
                with cols[1]:
                    x0_value = metrics.get('x0', metrics.get('punto_inflexion', 0))
                    st.metric("Punto de inflexión", f"{x0_value:.1f}")
                with cols[2]:
                    fase = metrics.get("Fase", metrics.get("fase", "No disponible"))
                    st.metric("Fase", fase)
        
        # Mostrar gráfico con datos
        if analysis.get("paper_data") or analysis.get("patent_data"):
            st.subheader("Visualización")
            
            fig = go.Figure()
            
            if analysis.get("paper_data"):
                paper_data = analysis["paper_data"]
                years = [int(year) for year in paper_data.keys()]
                values = list(paper_data.values())
                cumulative = []
                total = 0
                for v in values:
                    total += v
                    cumulative.append(total)
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=cumulative,
                    mode='lines+markers',
                    name='Papers (Acumulados)',
                    line=dict(color='blue', width=2)
                ))
            
            if analysis.get("patent_data"):
                patent_data = analysis["patent_data"]
                years = [int(year) for year in patent_data.keys()]
                values = list(patent_data.values())
                cumulative = []
                total = 0
                for v in values:
                    total += v
                    cumulative.append(total)
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=cumulative,
                    mode='lines+markers',
                    name='Patentes (Acumuladas)',
                    line=dict(color='red', width=2)
                ))
            
            fig.update_layout(
                title="Datos Acumulados",
                xaxis_title="Año",
                yaxis_title="Cantidad Acumulada",
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )
            
            st.plotly_chart(fig, use_container_width=True)

def manage_categories(db):
    """Interfaz para gestionar categorías en DynamoDB."""
    st.header("📂 Gestión de Categorías en DynamoDB")
    
    try:
        categories = db.get_all_categories()
    except Exception as e:
        st.error(f"Error al cargar categorías: {str(e)}")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Categorías Existentes")
        
        if not categories:
            st.info("No hay categorías definidas.")
        else:
            cat_data = []
            for cat in categories:
                cat_id = cat.get("category_id")
                
                try:
                    analyses_count = len(db.get_category_analysis(cat_id))
                except Exception:
                    analyses_count = 0
                
                created_at = cat.get("created_at", datetime.now().isoformat())
                try:
                    formatted_date = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime("%d/%m/%Y")
                except Exception:
                    formatted_date = str(created_at) if created_at else "Desconocida"
                
                cat_data.append({
                    "ID": cat_id or "Sin ID",
                    "Nombre": cat.get("name", "Sin nombre"),
                    "Descripción": cat.get("description", ""),
                    "Análisis": analyses_count,
                    "Creada": formatted_date
                })
            
            df_categories = pd.DataFrame(cat_data)
            st.dataframe(df_categories, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Crear Nueva Categoría")
        
        with st.form("new_category_form_dynamodb"):
            cat_name = st.text_input("Nombre de la categoría")
            cat_description = st.text_area("Descripción (opcional)")
            
            submit = st.form_submit_button("Crear Categoría")
            
            if submit and cat_name:
                try:
                    new_cat_id = db.create_category(cat_name, cat_description)
                    
                    if new_cat_id:
                        st.success(f"✅ Categoría '{cat_name}' creada con éxito.")
                        st.rerun()
                    else:
                        st.error("❌ Error al crear la categoría.")
                except Exception as e:
                    st.error(f"❌ Error al crear categoría: {str(e)}")

def comparative_analysis(db):
    """Interfaz para realizar análisis comparativos entre diferentes búsquedas guardadas."""
    st.header("📈 Análisis Comparativo en DynamoDB")
    st.write("""
    Compara diferentes análisis de curvas S guardados en DynamoDB para identificar 
    tendencias y relaciones entre diferentes tecnologías.
    """)
    
    try:
        all_analyses = db.storage.get_all_searches()
    except Exception as e:
        st.error(f"Error al cargar análisis: {str(e)}")
        return
    
    if not all_analyses:
        st.info("No hay análisis guardados para comparar.")
        return
    
    # Filtrar para mostrar solo los que tienen datos
    valid_analyses = [
        a for a in all_analyses 
        if a.get("paper_data") or a.get("patent_data")
    ]
    
    if not valid_analyses:
        st.warning("Los análisis guardados no contienen datos suficientes para comparar.")
        return
    
    # Opciones de selección
    st.subheader("Seleccionar Análisis a Comparar")
    
    try:
        categories = db.get_all_categories()
    except Exception:
        categories = [{"category_id": "default", "name": "Sin categoría"}]
    
    # Crear estructura jerárquica de selección
    category_analyses = {}
    
    for cat in categories:
        cat_id = cat.get("category_id")
        cat_name = cat.get("name", "Sin nombre")
        
        cat_analyses = [a for a in valid_analyses if a.get("category_id") == cat_id]
        
        if cat_analyses:
            analysis_dict = {}
            for i, analysis in enumerate(cat_analyses):
                analysis_id = analysis.get("analysis_id") or analysis.get("id")
                analysis_name = analysis.get("name") or f"Análisis {i+1}"
                analysis_dict[analysis_name] = analysis_id
            
            category_analyses[cat_name] = analysis_dict
    
    # Widget de multiselección
    selected_analyses = []
    
    for cat_name, analyses in category_analyses.items():
        st.write(f"**Categoría: {cat_name}**")
        
        for analysis_name, analysis_id in analyses.items():
            if st.checkbox(analysis_name, key=f"check_{analysis_id}_dynamo"):
                selected_analyses.append(analysis_id)
    
    if not selected_analyses:
        st.info("Selecciona al menos un análisis para continuar.")
        return
    
    # Opciones de visualización
    st.subheader("Opciones de Visualización")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_type = st.radio(
            "Tipo de datos a comparar",
            options=["Papers", "Patentes", "Ambos"],
            index=2
        )
    
    with col2:
        view_mode = st.radio(
            "Modo de visualización",
            options=["Acumulativo", "Anual", "Ambos"],
            index=0
        )
    
    # Botón para generar comparación
    if st.button("Generar Comparación", type="primary"):
        with st.spinner("Generando análisis comparativo..."):
            comparison_data = []
            
            for analysis_id in selected_analyses:
                try:
                    analysis = db.get_analysis_by_id(analysis_id)
                    
                    if not analysis:
                        continue
                    
                    analysis_name = analysis.get("name") or f"Análisis {analysis_id}"
                    
                    if data_type in ["Papers", "Ambos"] and analysis.get("paper_data"):
                        paper_data = analysis["paper_data"]
                        
                        years = [int(y) for y in paper_data.keys()]
                        values = list(paper_data.values())
                        
                        cum_values = []
                        cum_total = 0
                        for v in values:
                            cum_total += v
                            cum_values.append(cum_total)
                        
                        comparison_data.append({
                            "id": analysis_id,
                            "name": f"{analysis_name} (Papers)",
                            "years": years,
                            "values": values,
                            "cumulative": cum_values,
                            "type": "Papers",
                            "metrics": analysis.get("paper_metrics", {})
                        })
                    
                    if data_type in ["Patentes", "Ambos"] and analysis.get("patent_data"):
                        patent_data = analysis["patent_data"]
                        
                        years = [int(y) for y in patent_data.keys()]
                        values = list(patent_data.values())
                        
                        cum_values = []
                        cum_total = 0
                        for v in values:
                            cum_total += v
                            cum_values.append(cum_total)
                        
                        comparison_data.append({
                            "id": analysis_id,
                            "name": f"{analysis_name} (Patentes)",
                            "years": years,
                            "values": values,
                            "cumulative": cum_values,
                            "type": "Patentes",
                            "metrics": analysis.get("patent_metrics", {})
                        })
                
                except Exception as e:
                    st.warning(f"Error procesando análisis {analysis_id}: {str(e)}")
                    continue
            
            if not comparison_data:
                st.warning("No hay datos disponibles para los análisis seleccionados.")
                return
            
            # Mostrar gráficos según modo de visualización
            if view_mode in ["Acumulativo", "Ambos"]:
                st.subheader("Comparación de Curvas Acumulativas")
                
                fig_cum = go.Figure()
                
                for data in comparison_data:
                    fig_cum.add_trace(go.Scatter(
                        x=data["years"],
                        y=data["cumulative"],
                        mode='lines+markers',
                        name=data["name"],
                        hovertemplate="%{y} documentos en %{x}"
                    ))
                
                fig_cum.update_layout(
                    title="Comparación de Curvas S (Acumulativo)",
                    xaxis_title="Año",
                    yaxis_title="Cantidad Acumulada",
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                )
                
                st.plotly_chart(fig_cum, use_container_width=True)
            
            if view_mode in ["Anual", "Ambos"]:
                st.subheader("Comparación de Publicaciones Anuales")
                
                fig_annual = go.Figure()
                
                for data in comparison_data:
                    fig_annual.add_trace(go.Bar(
                        x=data["years"],
                        y=data["values"],
                        name=data["name"],
                        hovertemplate="%{y} documentos en %{x}"
                    ))
                
                fig_annual.update_layout(
                    title="Comparación de Publicaciones Anuales",
                    xaxis_title="Año",
                    yaxis_title="Cantidad Anual",
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray'),
                    barmode='group'
                )
                
                st.plotly_chart(fig_annual, use_container_width=True)