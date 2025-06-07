# utils/scopus_builder.py
import streamlit as st
import re
from typing import List, Dict, Tuple

def scopus_equation_interface():
    """
    Interfaz interactiva para construir ecuaciones de búsqueda en Scopus
    
    Returns:
        str: Ecuación de búsqueda formateada para Scopus
    """
    st.subheader("🔧 Constructor de Ecuaciones Scopus")
    
    # Inicializar estado si no existe
    if 'scopus_terms' not in st.session_state:
        st.session_state.scopus_terms = [{"field": "TITLE", "value": "", "operator": "AND"}]
    
    # Mostrar guía de campos de Scopus
    with st.expander("📖 Guía de Campos de Scopus", expanded=False):
        st.markdown("""
        ### Campos Principales de Búsqueda:
        - **TITLE**: Título del documento
        - **ABS**: Abstract/Resumen
        - **KEY**: Palabras clave del autor
        - **TITLE-ABS-KEY**: Título, resumen y palabras clave combinados
        - **AUTH**: Autor
        - **AFFIL**: Afiliación
        - **PUBYEAR**: Año de publicación
        - **DOCTYPE**: Tipo de documento (ar=artículo, re=review, cp=conference paper)
        - **SUBJAREA**: Área temática
        - **LANGUAGE**: Idioma
        
        ### Operadores:
        - **AND**: Ambos términos deben estar presentes
        - **OR**: Cualquiera de los términos debe estar presente
        - **AND NOT**: Excluir términos
        
        ### Ejemplos:
        - `TITLE("machine learning") AND KEY("agriculture")`
        - `TITLE-ABS-KEY("blockchain") AND PUBYEAR > 2020`
        - `AUTH("Smith") AND AFFIL("MIT")`
        """)
    
    # Constructor de términos
    st.write("### 🔍 Construir Ecuación")
    
    terms_to_remove = []
    
    for i, term in enumerate(st.session_state.scopus_terms):
        col1, col2, col3, col4, col5 = st.columns([2, 3, 2, 1, 1])
        
        with col1:
            # Selector de campo
            field_options = [
                "TITLE", "ABS", "KEY", "TITLE-ABS-KEY", "AUTH", "AFFIL", 
                "PUBYEAR", "DOCTYPE", "SUBJAREA", "LANGUAGE", "ALL"
            ]
            
            field = st.selectbox(
                f"Campo {i+1}",
                options=field_options,
                index=field_options.index(term["field"]) if term["field"] in field_options else 0,
                key=f"scopus_field_{i}"
            )
            term["field"] = field
        
        with col2:
            # Valor del término
            value = st.text_input(
                f"Valor {i+1}",
                value=term["value"],
                key=f"scopus_value_{i}",
                placeholder="Ej: machine learning, 2020, Smith"
            )
            term["value"] = value
        
        with col3:
            # Operador (solo si no es el último término)
            if i < len(st.session_state.scopus_terms) - 1:
                operator = st.selectbox(
                    f"Operador {i+1}",
                    options=["AND", "OR", "AND NOT"],
                    index=["AND", "OR", "AND NOT"].index(term["operator"]) if term["operator"] in ["AND", "OR", "AND NOT"] else 0,
                    key=f"scopus_operator_{i}"
                )
                term["operator"] = operator
            else:
                st.write("—")
        
        with col4:
            # Checkbox para comillas exactas
            exact_match = st.checkbox(
                "\"Exacto\"",
                value=term.get("exact_match", False),
                key=f"scopus_exact_{i}"
            )
            term["exact_match"] = exact_match
        
        with col5:
            # Botón para eliminar término
            if len(st.session_state.scopus_terms) > 1:
                if st.button("❌", key=f"scopus_remove_{i}"):
                    terms_to_remove.append(i)
    
    # Eliminar términos marcados
    if terms_to_remove:
        for idx in sorted(terms_to_remove, reverse=True):
            del st.session_state.scopus_terms[idx]
        st.rerun()
    
    # Botones de control
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("➕ Añadir Término"):
            st.session_state.scopus_terms.append({
                "field": "TITLE",
                "value": "",
                "operator": "AND",
                "exact_match": False
            })
            st.rerun()
    
    with col2:
        if st.button("🗑️ Limpiar Todo"):
            st.session_state.scopus_terms = [{"field": "TITLE", "value": "", "operator": "AND"}]
            st.rerun()
    
    # Construir y mostrar ecuación
    equation = build_scopus_equation(st.session_state.scopus_terms)
    
    if equation:
        st.write("### 📝 Ecuación Generada:")
        st.code(equation, language="text")
        
        # Validar ecuación
        is_valid, validation_message = validate_scopus_equation(equation)
        
        if is_valid:
            st.success(f"✅ {validation_message}")
        else:
            st.error(f"❌ {validation_message}")
        
        # Botón para usar ecuación predefinida
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Copiar Ecuación"):
                st.code(equation)
                st.success("Ecuación lista para copiar")
        
        with col2:
            if st.button("🔄 Cargar Ejemplos"):
                _load_example_equations()
        
        return equation
    else:
        st.info("Añade al menos un término con valor para generar la ecuación")
        return ""

def build_scopus_equation(terms: List[Dict]) -> str:
    """
    Construye la ecuación de Scopus basada en los términos
    
    Args:
        terms: Lista de términos con campo, valor y operador
        
    Returns:
        str: Ecuación formateada para Scopus
    """
    equation_parts = []
    
    for i, term in enumerate(terms):
        if not term["value"].strip():
            continue
        
        field = term["field"]
        value = term["value"].strip()
        
        # Aplicar comillas si es necesario
        if term.get("exact_match", False) and not (value.startswith('"') and value.endswith('"')):
            value = f'"{value}"'
        
        # Construir el término con su campo
        if field == "ALL":
            term_part = value
        else:
            term_part = f'{field}({value})'
        
        equation_parts.append(term_part)
        
        # Añadir operador si no es el último término
        if i < len(terms) - 1 and i < len([t for t in terms if t["value"].strip()]) - 1:
            equation_parts.append(term["operator"])
    
    return " ".join(equation_parts)

def validate_scopus_equation(equation: str) -> Tuple[bool, str]:
    """
    Valida una ecuación de Scopus
    
    Args:
        equation: Ecuación a validar
        
    Returns:
        Tuple[bool, str]: (es_válida, mensaje)
    """
    if not equation.strip():
        return False, "La ecuación está vacía"
    
    # Verificar paréntesis balanceados
    if equation.count("(") != equation.count(")"):
        return False, "Paréntesis no están balanceados"
    
    # Verificar comillas balanceadas
    if equation.count('"') % 2 != 0:
        return False, "Comillas no están balanceadas"
    
    # Verificar que no termine con operador
    if equation.strip().endswith(("AND", "OR", "AND NOT")):
        return False, "La ecuación no puede terminar con un operador"
    
    # Verificar campos válidos
    valid_fields = [
        "TITLE", "ABS", "KEY", "TITLE-ABS-KEY", "AUTH", "AUTHFIRST", "AUTHLAST",
        "AFFIL", "PUBYEAR", "DOCTYPE", "SUBJAREA", "LANGUAGE", "PMID", "DOI",
        "ISSN", "ISBN", "VOLUME", "ISSUE", "PAGES", "ARTNUM", "SRCTYPE",
        "CONF", "REFTEXT", "CHEM", "CAS", "FUND", "OPENACCESS"
    ]
    
    # Extraer campos usados en la ecuación
    field_pattern = r'([A-Z-]+)\('
    used_fields = re.findall(field_pattern, equation)
    
    invalid_fields = [field for field in used_fields if field not in valid_fields]
    if invalid_fields:
        return False, f"Campos inválidos encontrados: {', '.join(invalid_fields)}"
    
    # Verificar sintaxis básica
    if " AND AND " in equation or " OR OR " in equation:
        return False, "Operadores duplicados encontrados"
    
    return True, "Ecuación válida"

def parse_scopus_query(equation: str) -> List[Dict]:
    """
    Parsea una ecuación de Scopus existente en términos individuales
    
    Args:
        equation: Ecuación de Scopus
        
    Returns:
        List[Dict]: Lista de términos parseados
    """
    terms = []
    
    try:
        # Dividir por operadores principales
        parts = re.split(r'\s+(AND|OR|AND NOT)\s+', equation)
        
        for i in range(0, len(parts), 2):  # Saltar operadores
            part = parts[i].strip()
            
            # Extraer campo y valor
            if "(" in part and ")" in part:
                field_match = re.match(r'([A-Z-]+)\((.*)\)', part)
                if field_match:
                    field = field_match.group(1)
                    value = field_match.group(2)
                    
                    # Quitar comillas si existen
                    exact_match = value.startswith('"') and value.endswith('"')
                    if exact_match:
                        value = value[1:-1]
                    
                    # Determinar operador (si existe)
                    operator = "AND"
                    if i + 1 < len(parts):
                        operator = parts[i + 1]
                    
                    terms.append({
                        "field": field,
                        "value": value,
                        "operator": operator,
                        "exact_match": exact_match
                    })
                else:
                    # Si no se puede parsear, tratarlo como término general
                    terms.append({
                        "field": "ALL",
                        "value": part,
                        "operator": "AND" if i + 1 < len(parts) else "",
                        "exact_match": False
                    })
            else:
                # Término sin campo específico
                terms.append({
                    "field": "ALL",
                    "value": part,
                    "operator": "AND" if i + 1 < len(parts) else "",
                    "exact_match": False
                })
    
    except Exception as e:
        st.warning(f"Error parseando ecuación: {str(e)}")
        # Retornar término básico si falla el parsing
        terms = [{
            "field": "TITLE-ABS-KEY",
            "value": equation,
            "operator": "AND",
            "exact_match": False
        }]
    
    return terms

def _load_example_equations():
    """Carga ecuaciones de ejemplo en el constructor"""
    examples = {
        "Inteligencia Artificial en Agricultura": [
            {"field": "TITLE-ABS-KEY", "value": "artificial intelligence", "operator": "AND", "exact_match": True},
            {"field": "TITLE-ABS-KEY", "value": "agriculture", "operator": "OR", "exact_match": False},
            {"field": "TITLE-ABS-KEY", "value": "farming", "operator": "AND", "exact_match": False},
            {"field": "PUBYEAR", "value": "> 2020", "operator": "AND", "exact_match": False}
        ],
        "Blockchain en Finanzas": [
            {"field": "TITLE", "value": "blockchain", "operator": "AND", "exact_match": False},
            {"field": "TITLE-ABS-KEY", "value": "finance", "operator": "OR", "exact_match": False},
            {"field": "TITLE-ABS-KEY", "value": "financial", "operator": "AND NOT", "exact_match": False},
            {"field": "DOCTYPE", "value": "ar", "operator": "", "exact_match": False}
        ],
        "Machine Learning en Medicina": [
            {"field": "TITLE-ABS-KEY", "value": "machine learning", "operator": "AND", "exact_match": True},
            {"field": "TITLE-ABS-KEY", "value": "medical", "operator": "OR", "exact_match": False},
            {"field": "TITLE-ABS-KEY", "value": "healthcare", "operator": "", "exact_match": False}
        ]
    }
    
    st.write("### 📋 Ejemplos de Ecuaciones")
    
    selected_example = st.selectbox(
        "Selecciona un ejemplo:",
        options=list(examples.keys())
    )
    
    if st.button("Cargar Ejemplo Seleccionado"):
        st.session_state.scopus_terms = examples[selected_example]
        st.success(f"✅ Ejemplo '{selected_example}' cargado")
        st.rerun()
    
    # Mostrar preview del ejemplo seleccionado
    example_equation = build_scopus_equation(examples[selected_example])
    st.code(example_equation, language="text")

def export_scopus_equation(equation: str, terms: List[Dict]) -> Dict:
    """
    Exporta la ecuación y términos en formato reutilizable
    
    Args:
        equation: Ecuación construida
        terms: Términos utilizados
        
    Returns:
        Dict: Datos exportables
    """
    export_data = {
        "equation": equation,
        "terms": terms,
        "created_at": pd.Timestamp.now().isoformat(),
        "version": "1.0",
        "description": "Ecuación de Scopus generada por Tech Trends Explorer"
    }
    
    return export_data

def import_scopus_equation(import_data: Dict) -> bool:
    """
    Importa una ecuación previamente exportada
    
    Args:
        import_data: Datos de ecuación exportada
        
    Returns:
        bool: Éxito de la importación
    """
    try:
        if "terms" in import_data:
            st.session_state.scopus_terms = import_data["terms"]
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error importando ecuación: {str(e)}")
        return False

def get_scopus_field_suggestions(field_type: str) -> List[str]:
    """
    Proporciona sugerencias para diferentes tipos de campos
    
    Args:
        field_type: Tipo de campo de Scopus
        
    Returns:
        List[str]: Lista de sugerencias
    """
    suggestions = {
        "TITLE": [
            "artificial intelligence", "machine learning", "blockchain", 
            "quantum computing", "internet of things", "big data"
        ],
        "DOCTYPE": [
            "ar (article)", "re (review)", "cp (conference paper)", 
            "ch (book chapter)", "bk (book)", "le (letter)"
        ],
        "SUBJAREA": [
            "COMP (Computer Science)", "ENGI (Engineering)", "MEDI (Medicine)",
            "PHYS (Physics)", "CHEM (Chemistry)", "MATH (Mathematics)"
        ],
        "LANGUAGE": [
            "english", "spanish", "french", "german", "chinese", "japanese"
        ],
        "PUBYEAR": [
            "> 2020", "= 2023", "< 2022", "> 2015 AND < 2025"
        ]
    }
    
    return suggestions.get(field_type, [])

# Funciones adicionales para análisis avanzado
def optimize_scopus_query(equation: str) -> Tuple[str, List[str]]:
    """
    Optimiza una ecuación de Scopus para mejor rendimiento
    
    Args:
        equation: Ecuación original
        
    Returns:
        Tuple[str, List[str]]: (ecuación_optimizada, sugerencias)
    """
    optimized = equation
    suggestions = []
    
    # Optimización 1: Usar TITLE-ABS-KEY en lugar de campos separados
    if "TITLE(" in equation and "ABS(" in equation and "KEY(" in equation:
        suggestions.append("Considera usar TITLE-ABS-KEY para combinar título, resumen y palabras clave")
    
    # Optimización 2: Añadir filtros de tipo de documento
    if "DOCTYPE" not in equation:
        suggestions.append("Considera añadir DOCTYPE(ar) para limitar a artículos de investigación")
    
    # Optimización 3: Añadir filtros temporales
    if "PUBYEAR" not in equation:
        suggestions.append("Considera añadir filtros de año para resultados más relevantes")
    
    # Optimización 4: Simplificar operadores complejos
    if " AND NOT " in equation:
        suggestions.append("Los operadores AND NOT pueden ser costosos, considera reformular")
    
    return optimized, suggestions