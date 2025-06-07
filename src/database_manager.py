# src/database_manager.py
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
    Ejecuta la interfaz de gestión de la base de datos.
    Permite ver, organizar y analizar los datos guardados.
    
    Args:
        db: Instancia de SCurveDatabase (opcional)
    """
    st.title("📊 Gestor de Datos Guardados")
    
    # Si no se proporciona db, intentar inicializar con almacenamiento local
    if db is None:
        try:
            db = initialize_database("local")
        except Exception as e:
            st.error(f"Error al inicializar base de datos: {str(e)}")
            st.info("No se pudo inicializar el sistema de almacenamiento")
            return
    
    if db is None:
        st.error("No se pudo inicializar el sistema de almacenamiento.")
        return
    
    # Mostrar tipo de almacenamiento activo
    storage_type = "DynamoDB" if hasattr(db.storage, 'dynamodb') else "Local"
    st.info(f"🗄️ Almacenamiento activo: **{storage_type}**")
    
    # Opciones de la interfaz
    tab1, tab2, tab3 = st.tabs([
        "🔍 Ver Análisis Guardados", 
        "📂 Gestionar Categorías", 
        "📈 Análisis Comparativo"
    ])
    
    with tab1:
        show_saved_analyses(db)
    
    with tab2:
        manage_categories(db)
    
    with tab3:
        comparative_analysis(db)

def show_saved_analyses(db):
    """
    Muestra todos los análisis guardados con opciones de filtrado.
    
    Args:
        db: Instancia de SCurveDatabase
    """
    st.header("🔍 Análisis Guardados")
    
    # Obtener categorías y análisis
    try:
        categories = db.get_all_categories()
        all_analyses = db.storage.get_all_searches()
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return
    
    # Si no hay datos, mostrar mensaje
    if not all_analyses:
        st.info("No hay análisis guardados. Realiza algunas búsquedas en las pestañas de análisis.")
        return
    
    # Opciones de filtrado
    st.subheader("Filtrar Análisis")
    
    # Por categoría - KEY ÚNICA AÑADIDA
    category_options = {"Todas las categorías": "all"}
    category_options.update({cat["name"]: cat["id"] for cat in categories})
    
    selected_category = st.selectbox(
        "Categoría",
        options=list(category_options.keys()),
        index=0,
        key="db_manager_category_filter_selectbox"  # ← KEY ÚNICA AÑADIDA
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
        key="db_manager_date_range_input"  # ← KEY ÚNICA AÑADIDA
    )
    
    # Aplicar filtros
    filtered_analyses = []
    
    for analysis in all_analyses:
        # Filtrar por categoría
        if selected_category_id != "all" and analysis.get("category_id") != selected_category_id:
            continue
            
        # Filtrar por fecha
        try:
            # Manejar diferentes formatos de timestamp
            timestamp = analysis.get("timestamp") or analysis.get("execution_date")
            if timestamp:
                analysis_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                if len(date_range) == 2:
                    if not (date_range[0] <= analysis_date <= date_range[1]):
                        continue
        except Exception:
            # Si no se puede parsear la fecha, incluir el análisis
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
            if cat.get("id") == analysis.get("category_id") or cat.get("category_id") == analysis.get("category_id"):
                category_name = cat.get("name", "Sin nombre")
                break
        
        # Manejar diferentes estructuras de ID
        analysis_id = analysis.get("id") or analysis.get("analysis_id")
        
        # Manejar diferentes formatos de timestamp
        timestamp = analysis.get("timestamp") or analysis.get("execution_date")
        try:
            if timestamp:
                formatted_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime("%d/%m/%Y %H:%M")
            else:
                formatted_date = "Fecha desconocida"
        except Exception:
            formatted_date = str(timestamp) if timestamp else "Fecha desconocida"
        
        # Extraer datos básicos
        analysis_data = {
            "ID": analysis_id or "Sin ID",
            "Nombre": analysis.get("name") or "Sin nombre",
            "Categoría": category_name,
            "Fecha": formatted_date,
            "Consulta": analysis.get("query") or "Sin consulta",
            "Tipo": analysis.get("analysis_type") or "Desconocido",
            "Datos de Papers": "Sí" if analysis.get("paper_data") else "No",
            "Datos de Patentes": "Sí" if analysis.get("patent_data") else "No"
        }
        
        analyses_data.append(analysis_data)
    
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
            analysis_id = analysis.get("id") or analysis.get("analysis_id")
            analysis_name = analysis.get("name") or f"Análisis {analysis_id}"
            analysis_options[analysis_name] = analysis_id
        
        # Selectbox para elegir análisis - KEY ÚNICA AÑADIDA
        selected_analysis_name = st.selectbox(
            "Selecciona un análisis para ver detalles",
            options=list(analysis_options.keys()),
            key="db_manager_analysis_details_selectbox"  # ← KEY ÚNICA AÑADIDA
        )
        
        selected_analysis_id = analysis_options[selected_analysis_name]
        
        if selected_analysis_id:
            display_analysis_details(db, selected_analysis_id)

def display_analysis_details(db, analysis_id):
    """
    Muestra los detalles de un análisis específico.
    
    Args:
        db: Instancia de SCurveDatabase
        analysis_id: ID del análisis a mostrar
    """
    # Obtener datos del análisis
    try:
        analysis = db.get_analysis_by_id(analysis_id)
    except Exception as e:
        st.error(f"Error al obtener análisis: {str(e)}")
        return
    
    if not analysis:
        st.error(f"No se encontró el análisis con ID: {analysis_id}")
        return
    
    # Crear contenedor expandible
    analysis_name = analysis.get('name') or 'Análisis sin nombre'
    with st.expander(f"Detalles: {analysis_name}", expanded=True):
        # Información básica
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Consulta:**", analysis.get("query", "No especificada"))
            
            # Manejar diferentes formatos de timestamp
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
            # Categoría
            try:
                category = db.storage.get_category_by_id(analysis.get("category_id"))
                category_name = category.get("name") if category else "Desconocida"
            except Exception:
                category_name = "Error al obtener categoría"
            
            st.write("**Categoría:**", category_name)
            st.write("**Tipo de análisis:**", analysis.get("analysis_type", "No especificado"))
            
            # Conteo de datos
            paper_count = len(analysis.get("paper_data", {}))
            patent_count = len(analysis.get("patent_data", {}))
            st.write(f"**Datos:** {paper_count} años de papers, {patent_count} años de patentes")
        
        # Mostrar datos en tablas si existen
        if analysis.get("paper_data"):
            st.subheader("Datos de Papers")
            
            # Convertir a DataFrame
            paper_data = analysis["paper_data"]
            df_papers = pd.DataFrame({
                "Año": list(paper_data.keys()),
                "Cantidad": list(paper_data.values())
            })
            df_papers["Acumulado"] = df_papers["Cantidad"].cumsum()
            
            # Mostrar tabla
            st.dataframe(df_papers, use_container_width=True)
            
            # Mostrar métrica principal si existe
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
            
            # Convertir a DataFrame
            patent_data = analysis["patent_data"]
            df_patents = pd.DataFrame({
                "Año": list(patent_data.keys()),
                "Cantidad": list(patent_data.values())
            })
            df_patents["Acumulado"] = df_patents["Cantidad"].cumsum()
            
            # Mostrar tabla
            st.dataframe(df_patents, use_container_width=True)
            
            # Mostrar métrica principal si existe
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
            
            # Crear gráfico
            fig = go.Figure()
            
            # Añadir datos de papers si existen
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
            
            # Añadir datos de patentes si existen
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
            
            # Configurar aspecto del gráfico
            fig.update_layout(
                title="Datos Acumulados",
                xaxis_title="Año",
                yaxis_title="Cantidad Acumulada",
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )
            
            # Mostrar gráfico
            st.plotly_chart(fig, use_container_width=True)

def manage_categories(db):
    """
    Interfaz para gestionar categorías.
    
    Args:
        db: Instancia de SCurveDatabase
    """
    st.header("📂 Gestión de Categorías")
    
    # Obtener categorías
    try:
        categories = db.get_all_categories()
    except Exception as e:
        st.error(f"Error al cargar categorías: {str(e)}")
        return
    
    # Dividir en columnas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Mostrar categorías existentes
        st.subheader("Categorías Existentes")
        
        if not categories:
            st.info("No hay categorías definidas.")
        else:
            # Crear tabla
            cat_data = []
            for cat in categories:
                # Manejar diferentes estructuras de ID
                cat_id = cat.get("id") or cat.get("category_id")
                
                # Contar análisis en esta categoría
                try:
                    analyses_count = len(db.get_category_analysis(cat_id))
                except Exception:
                    analyses_count = 0
                
                # Manejar fecha de creación
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
            
            # Mostrar tabla
            df_categories = pd.DataFrame(cat_data)
            st.dataframe(
                df_categories,
                use_container_width=True,
                hide_index=True
            )
    
    with col2:
        # Formulario para crear nueva categoría
        st.subheader("Crear Nueva Categoría")
        
        with st.form("new_category_form_unique"):  # ← KEY ÚNICA AÑADIDA
            cat_name = st.text_input("Nombre de la categoría")
            cat_description = st.text_area("Descripción (opcional)")
            
            submit = st.form_submit_button("Crear Categoría")
            
            if submit and cat_name:
                try:
                    # Crear nueva categoría
                    new_cat_id = db.create_category(cat_name, cat_description)
                    
                    if new_cat_id:
                        st.success(f"✅ Categoría '{cat_name}' creada con éxito.")
                        st.rerun()
                    else:
                        st.error("❌ Error al crear la categoría.")
                except Exception as e:
                    st.error(f"❌ Error al crear categoría: {str(e)}")
    
    # Sección para reasignar análisis a categorías
    st.subheader("Reasignar Análisis")
    
    # Obtener todos los análisis
    try:
        all_analyses = db.storage.get_all_searches()
    except Exception as e:
        st.error(f"Error al cargar análisis: {str(e)}")
        return
    
    if not all_analyses:
        st.info("No hay análisis guardados para reasignar.")
        return
    
    # Opciones de selección
    analysis_options = {}
    for analysis in all_analyses:
        analysis_id = analysis.get("id") or analysis.get("analysis_id")
        analysis_name = analysis.get("name") or f"Análisis {analysis_id}"
        analysis_options[analysis_name] = analysis_id
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selectbox para análisis - KEY ÚNICA AÑADIDA
        selected_analysis_name = st.selectbox(
            "Selecciona un análisis",
            options=list(analysis_options.keys()),
            key="db_manager_reassign_analysis_selectbox"  # ← KEY ÚNICA AÑADIDA
        )
        selected_analysis_id = analysis_options[selected_analysis_name]
    
    with col2:
        # Selector de categoría destino - KEY ÚNICA AÑADIDA
        category_mapping = {}
        for cat in categories:
            cat_id = cat.get("id") or cat.get("category_id")
            cat_name = cat.get("name", "Sin nombre")
            category_mapping[cat_name] = cat_id
        
        selected_category = st.selectbox(
            "Mover a categoría",
            options=list(category_mapping.keys()),
            key="db_manager_reassign_category_selectbox"  # ← KEY ÚNICA AÑADIDA
        )
        selected_category_id = category_mapping[selected_category]
    
    # Botón para reasignar - KEY ÚNICA AÑADIDA
    if st.button("Reasignar Análisis", type="primary", key="db_manager_reassign_button"):
        try:
            # Obtener el análisis seleccionado
            analysis = db.get_analysis_by_id(selected_analysis_id)
            
            if analysis:
                current_category = analysis.get("category_id")
                if current_category != selected_category_id:
                    # Actualizar categoría
                    analysis["category_id"] = selected_category_id
                    
                    # Guardar cambios (esto depende del tipo de storage)
                    try:
                        # Para DynamoDB, necesitamos actualizar el item
                        if hasattr(db.storage, 'analyses_table'):
                            # DynamoDB update
                            db.storage.analyses_table.update_item(
                                Key={
                                    'analysis_id': analysis.get('analysis_id'),
                                    'timestamp': analysis.get('timestamp')
                                },
                                UpdateExpression='SET category_id = :cat_id',
                                ExpressionAttributeValues={':cat_id': selected_category_id}
                            )
                        else:
                            # Local storage update
                            db.storage.save_data()
                        
                        st.success(f"✅ Análisis movido a la categoría '{selected_category}'")
                        st.rerun()
                    except Exception as update_error:
                        st.error(f"❌ Error al actualizar: {str(update_error)}")
                else:
                    st.info("El análisis ya pertenece a esta categoría.")
            else:
                st.error("No se pudo encontrar el análisis seleccionado.")
        except Exception as e:
            st.error(f"❌ Error en la reasignación: {str(e)}")

def comparative_analysis(db):
    """
    Interfaz para realizar análisis comparativos entre diferentes búsquedas guardadas.
    
    Args:
        db: Instancia de SCurveDatabase
    """
    st.header("📈 Análisis Comparativo")
    st.write("""
    En esta sección puedes comparar diferentes análisis de curvas S guardados,
    para identificar tendencias y relaciones entre diferentes tecnologías.
    """)
    
    # Obtener todos los análisis
    try:
        all_analyses = db.storage.get_all_searches()
    except Exception as e:
        st.error(f"Error al cargar análisis: {str(e)}")
        return
    
    if not all_analyses:
        st.info("No hay análisis guardados para comparar. Realiza algunas búsquedas en las pestañas de análisis.")
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
    
    # Organizar por categorías
    try:
        categories = db.get_all_categories()
    except Exception:
        categories = [{"id": "default", "name": "Sin categoría"}]
    
    # Crear estructura jerárquica de selección
    category_analyses = {}
    
    for cat in categories:
        cat_id = cat.get("id") or cat.get("category_id")
        cat_name = cat.get("name", "Sin nombre")
        
        # Filtrar análisis para esta categoría
        cat_analyses = [a for a in valid_analyses if a.get("category_id") == cat_id]
        
        if cat_analyses:
            analysis_dict = {}
            for i, analysis in enumerate(cat_analyses):
                analysis_id = analysis.get("id") or analysis.get("analysis_id")
                analysis_name = analysis.get("name") or f"Análisis {i+1}"
                analysis_dict[analysis_name] = analysis_id
            
            category_analyses[cat_name] = analysis_dict
    
    # Widget de multiselección con estructura de árbol
    selected_analyses = []
    
    for cat_name, analyses in category_analyses.items():
        st.write(f"**Categoría: {cat_name}**")
        
        for analysis_name, analysis_id in analyses.items():
            if st.checkbox(analysis_name, key=f"check_{analysis_id}"):
                selected_analyses.append(analysis_id)
    
    # Continuar solo si hay análisis seleccionados
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
            # Recopilar datos de los análisis seleccionados
            comparison_data = []
            
            for analysis_id in selected_analyses:
                try:
                    analysis = db.get_analysis_by_id(analysis_id)
                    
                    if not analysis:
                        continue
                    
                    analysis_name = analysis.get("name") or f"Análisis {analysis_id}"
                    
                    # Procesar según tipo de datos seleccionado
                    if data_type in ["Papers", "Ambos"] and analysis.get("paper_data"):
                        paper_data = analysis["paper_data"]
                        
                        # Convertir a formato para gráfica
                        years = [int(y) for y in paper_data.keys()]
                        values = list(paper_data.values())
                        
                        # Calcular acumulado
                        cum_values = []
                        cum_total = 0
                        for v in values:
                            cum_total += v
                            cum_values.append(cum_total)
                        
                        # Añadir a datos de comparación
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
                        
                        # Convertir a formato para gráfica
                        years = [int(y) for y in patent_data.keys()]
                        values = list(patent_data.values())
                        
                        # Calcular acumulado
                        cum_values = []
                        cum_total = 0
                        for v in values:
                            cum_total += v
                            cum_values.append(cum_total)
                        
                        # Añadir a datos de comparación
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
            
            # Verificar si hay datos para comparar
            if not comparison_data:
                st.warning("No hay datos disponibles para los análisis seleccionados y tipo de datos.")
                return
            
            # Mostrar gráficos según modo de visualización
            if view_mode in ["Acumulativo", "Ambos"]:
                st.subheader("Comparación de Curvas Acumulativas")
                
                # Crear gráfico acumulativo
                fig_cum = go.Figure()
                
                for data in comparison_data:
                    fig_cum.add_trace(go.Scatter(
                        x=data["years"],
                        y=data["cumulative"],
                        mode='lines+markers',
                        name=data["name"],
                        hovertemplate="%{y} documentos en %{x}"
                    ))
                
                # Configurar aspecto del gráfico
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
                
                # Tabla con puntos de inflexión
                st.subheader("Puntos de Inflexión")
                
                inflection_data = []
                
                for data in comparison_data:
                    metrics = data["metrics"]
                    
                    if metrics and "x0" in metrics:
                        inflection_data.append({
                            "Análisis": data["name"],
                            "Punto de Inflexión": round(metrics["x0"], 1),
                            "R²": round(metrics.get("R2", 0), 4),
                            "Fase": metrics.get("Fase", "No disponible")
                        })
                
                if inflection_data:
                    df_inflection = pd.DataFrame(inflection_data)
                    st.dataframe(df_inflection, use_container_width=True)
                    
                    # Análisis de brecha temporal si hay múltiples curvas
                    if len(inflection_data) > 1:
                        st.subheader("Análisis de Brecha Temporal")
                        
                        # Si hay papers y patentes para el mismo análisis, mostrar time lag
                        for analysis_id in selected_analyses:
                            paper_data = None
                            patent_data = None
                            
                            for data in comparison_data:
                                if data["id"] == analysis_id:
                                    if data["type"] == "Papers":
                                        paper_data = data
                                    elif data["type"] == "Patentes":
                                        patent_data = data
                            
                            if paper_data and patent_data and "metrics" in paper_data and "metrics" in patent_data:
                                paper_metrics = paper_data["metrics"]
                                patent_metrics = patent_data["metrics"]
                                
                                if "x0" in paper_metrics and "x0" in patent_metrics:
                                    analysis_name = paper_data["name"].replace(" (Papers)", "")
                                    time_lag = patent_metrics["x0"] - paper_metrics["x0"]
                                    
                                    st.write(f"**{analysis_name}**: ")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric(
                                            "Punto de inflexión Papers", 
                                            f"{paper_metrics['x0']:.1f}"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            "Punto de inflexión Patentes", 
                                            f"{patent_metrics['x0']:.1f}"
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            "Time Lag (años)", 
                                            f"{time_lag:.1f}"
                                        )
                                    
                                    # Interpretación del time lag
                                    if time_lag > 0:
                                        st.info(f"Las patentes muestran un retraso de aproximadamente {time_lag:.1f} años respecto a las publicaciones académicas.")
                                    elif time_lag < 0:
                                        st.info(f"Las patentes muestran un adelanto de aproximadamente {abs(time_lag):.1f} años respecto a las publicaciones académicas.")
                                    else:
                                        st.info("No hay brecha temporal significativa entre publicaciones académicas y patentes.")
                else:
                    st.info("No hay datos de puntos de inflexión disponibles para los análisis seleccionados.")
            
            if view_mode in ["Anual", "Ambos"]:
                st.subheader("Comparación de Publicaciones Anuales")
                
                # Crear gráfico anual
                fig_annual = go.Figure()
                
                for data in comparison_data:
                    fig_annual.add_trace(go.Bar(
                        x=data["years"],
                        y=data["values"],
                        name=data["name"],
                        hovertemplate="%{y} documentos en %{x}"
                    ))
                
                # Configurar aspecto del gráfico
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