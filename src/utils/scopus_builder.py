# src/utils/scopus_builder.py
import streamlit as st
import re

def build_scopus_advanced_equation(term_groups):
    """
    Construye una ecuación de búsqueda Scopus avanzada basada en grupos de términos.
    
    Args:
        term_groups: Lista de diccionarios, cada uno con:
            - field: Campo de búsqueda (TITLE, ABSTRACT, etc.)
            - terms: Lista de términos para ese campo
            - operator: Operador para unir con el siguiente grupo
    
    Returns:
        String con la ecuación Scopus completa
    """
    parts = []
    
    for i, group in enumerate(term_groups):
        if not group['terms']:
            continue
            
        # Construir la parte del campo
        field = group['field']
        terms = group['terms']
        
        # Si hay múltiples términos, los unimos con OR
        if len(terms) > 1:
            terms_clause = " OR ".join([f'"{term}"' if ' ' in term else term for term in terms])
            terms_clause = f"({terms_clause})"
        else:
            term = terms[0]
            terms_clause = f'"{term}"' if ' ' in term else term
        
        # Agregar el campo con los términos
        field_clause = f"{field}({terms_clause})"
        
        # Si no es el último grupo y hay un operador, añadirlo
        if i < len(term_groups) - 1 and group.get('operator'):
            field_clause += f" {group['operator']}"
        
        parts.append(field_clause)
    
    # Unir todas las partes
    equation = " ".join(parts)
    return equation

def scopus_equation_interface():
    """
    Interfaz de usuario para construir ecuaciones Scopus avanzadas.
    
    Returns:
        String con la ecuación Scopus construida
    """
    st.write("### 🔍 Constructor de Ecuación Scopus")
    
    # Inicializar estado si no existe
    if 'scopus_term_groups' not in st.session_state:
        st.session_state.scopus_term_groups = [
            {
                'id': 0,
                'field': 'TITLE',
                'terms': [],
                'operator': 'AND'
            }
        ]
    
    # Funciones para manipular grupos de términos
    def add_term_group():
        """Añade un nuevo grupo de términos"""
        new_id = max([g['id'] for g in st.session_state.scopus_term_groups]) + 1
        st.session_state.scopus_term_groups.append({
            'id': new_id,
            'field': 'TITLE',
            'terms': [],
            'operator': 'AND'
        })
    
    def remove_term_group(group_id):
        """Elimina un grupo de términos"""
        st.session_state.scopus_term_groups = [
            g for g in st.session_state.scopus_term_groups 
            if g['id'] != group_id
        ]
    
    def add_term(group_id, term):
        """Añade un término a un grupo específico"""
        if not term:
            return
            
        for group in st.session_state.scopus_term_groups:
            if group['id'] == group_id:
                if term not in group['terms']:
                    group['terms'].append(term)
                break
    
    def remove_term(group_id, term):
        """Elimina un término de un grupo específico"""
        for group in st.session_state.scopus_term_groups:
            if group['id'] == group_id:
                group['terms'] = [t for t in group['terms'] if t != term]
                break
    
    def update_field(group_id, field):
        """Actualiza el campo de un grupo"""
        for group in st.session_state.scopus_term_groups:
            if group['id'] == group_id:
                group['field'] = field
                break
    
    def update_operator(group_id, operator):
        """Actualiza el operador de un grupo"""
        for group in st.session_state.scopus_term_groups:
            if group['id'] == group_id:
                group['operator'] = operator
                break
    
    # Mostrar interfaz para cada grupo de términos
    for group in st.session_state.scopus_term_groups:
        with st.expander(f"Grupo {group['id']+1}: {group['field']}", expanded=True):
            # Selección de campo
            col1, col2 = st.columns([3, 1])
            with col1:
                field = st.selectbox(
                    "Campo de búsqueda",
                    options=["TITLE", "TITLE-ABS-KEY", "AUTHOR", "AFFIL", "SRCTITLE"],
                    index=["TITLE", "TITLE-ABS-KEY", "AUTHOR", "AFFIL", "SRCTITLE"].index(group['field']),
                    key=f"field_{group['id']}",
                    help="Campo en el que se buscarán los términos"
                )
                update_field(group['id'], field)
            
            with col2:
                operator = st.selectbox(
                    "Operador",
                    options=["AND", "OR", "AND NOT"],
                    index=["AND", "OR", "AND NOT"].index(group['operator']),
                    key=f"operator_{group['id']}",
                    help="Operador para vincular con el siguiente grupo"
                )
                update_operator(group['id'], operator)
            
            # Mostrar términos actuales
            if group['terms']:
                st.write("**Términos añadidos:**")
                term_cols = st.columns(3)
                for i, term in enumerate(group['terms']):
                    col_idx = i % 3
                    with term_cols[col_idx]:
                        st.write(f"{i+1}. {term}")
                        if st.button("🗑️", key=f"remove_term_{group['id']}_{i}"):
                            remove_term(group['id'], term)
            
            # Entrada para nuevos términos
            new_term = st.text_input(
                "Añadir término",
                key=f"new_term_{group['id']}",
                placeholder="Ej: artificial intelligence"
            )
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("➕ Añadir", key=f"add_term_{group['id']}"):
                    add_term(group['id'], new_term)
                    # Limpiar el campo
                    st.session_state[f"new_term_{group['id']}"] = ""
            
            with col2:
                if st.button("📋 Ejemplos comunes", key=f"examples_{group['id']}"):
                    examples = {
                        "TITLE": ["enzyme", "glucose oxidase", "flour", "starch", "noodles", "pasta"],
                        "TITLE-ABS-KEY": ["gelatinization", "pulp", "antioxidant", "phenolic"]
                    }
                    field_examples = examples.get(group['field'], ["No hay ejemplos para este campo"])
                    
                    example_cols = st.columns(3)
                    for i, example in enumerate(field_examples):
                        col_idx = i % 3
                        with example_cols[col_idx]:
                            if st.button(example, key=f"example_{group['id']}_{i}"):
                                add_term(group['id'], example)
        
        # Botón para eliminar grupo
        col1, col2 = st.columns([3, 1])
        with col2:
            if len(st.session_state.scopus_term_groups) > 1:
                if st.button("🗑️ Eliminar grupo", key=f"remove_group_{group['id']}"):
                    remove_term_group(group['id'])
    
    # Botón para añadir nuevo grupo
    if st.button("➕ Añadir grupo de términos", key="add_group"):
        add_term_group()
    
    # Costruir y mostrar la ecuación completa
    equation = build_scopus_advanced_equation(st.session_state.scopus_term_groups)
    
    st.write("### 📝 Ecuación de búsqueda para Scopus")
    st.code(equation)
    
    # Sugerencias basadas en patrones comunes
    with st.expander("📚 Ejemplos de Ecuaciones Comunes"):
        examples = [
            'TITLE("Plantain" OR "banana" OR "musa") AND TITLE("flour" OR "starch") AND TITLE("enzyme" OR "glucose oxidase")',
            'TITLE("Plantain" OR "banana" OR "musa") AND TITLE("noodles" OR "pasta")',
            'TITLE("Plantain" OR "banana" OR "musa") AND TITLE("flour" OR "starch") AND TITLE-ABS-KEY(glucose AND oxidase)',
            'TITLE("noodles" OR "pasta" OR "raviolis") AND TITLE("egg" OR "gum") AND TITLE("free" AND "gluten")',
            'TITLE(plantain OR musa) AND TITLE-ABS-KEY(gelatinization AND pulp)'
        ]
        
        for i, example in enumerate(examples):
            st.markdown(f"**Ejemplo {i+1}**: `{example}`")
            if st.button(f"Usar este ejemplo", key=f"use_example_{i}"):
                # Parsear el ejemplo y configurar la interfaz
                parse_and_set_equation(example)
    
    return equation

def parse_and_set_equation(equation):
    """
    Parsea una ecuación Scopus y configura la interfaz para reflejarla.
    
    Args:
        equation: String con la ecuación Scopus a parsear
    """
    # Reset the current groups
    st.session_state.scopus_term_groups = []
    
    # Define regex pattern to extract FIELD(terms) AND/OR FIELD(terms)
    pattern = r'(TITLE|TITLE-ABS-KEY|AUTHOR|AFFIL|SRCTITLE)\(([^)]+)\)(?:\s+(AND|OR|AND NOT))?'
    
    matches = re.finditer(pattern, equation)
    
    group_id = 0
    for match in matches:
        field = match.group(1)
        terms_str = match.group(2)
        operator = match.group(3) if match.group(3) else "AND"
        
        # Extract individual terms, respecting quoted terms
        terms = []
        in_quotes = False
        current_term = ""
        
        i = 0
        while i < len(terms_str):
            char = terms_str[i]
            
            if char == '"':
                in_quotes = not in_quotes
                current_term += char
            elif char == ' ' and not in_quotes:
                if current_term.strip():
                    if current_term.lower().strip() not in ('and', 'or'):
                        terms.append(current_term.strip())
                current_term = ""
            elif (char == 'O' and i+1 < len(terms_str) and terms_str[i+1] == 'R' and 
                  (i+2 == len(terms_str) or terms_str[i+2].isspace()) and not in_quotes):
                # Skip "OR" operator
                i += 1  # Skip 'R'
                current_term = ""
            elif (char == 'A' and i+2 < len(terms_str) and terms_str[i+1] == 'N' and 
                  terms_str[i+2] == 'D' and (i+3 == len(terms_str) or terms_str[i+3].isspace()) and 
                  not in_quotes):
                # Skip "AND" operator
                i += 2  # Skip 'ND'
                current_term = ""
            else:
                current_term += char
            
            i += 1
        
        # Add the last term if any
        if current_term.strip():
            terms.append(current_term.strip())
        
        # Clean up terms - remove quotes
        cleaned_terms = []
        for term in terms:
            if term.startswith('"') and term.endswith('"'):
                cleaned_terms.append(term[1:-1])
            else:
                cleaned_terms.append(term)
        
        # Add the group
        st.session_state.scopus_term_groups.append({
            'id': group_id,
            'field': field,
            'terms': cleaned_terms,
            'operator': operator
        })
        
        group_id += 1
    
    # Ensure at least one group exists
    if not st.session_state.scopus_term_groups:
        st.session_state.scopus_term_groups = [{
            'id': 0,
            'field': 'TITLE',
            'terms': [],
            'operator': 'AND'
        }]

def parse_scopus_query(query):
    """
    Parsea una consulta de Scopus y la convierte a un formato estructurado.
    
    Args:
        query: String con la consulta Scopus
        
    Returns:
        Lista de diccionarios con los grupos de términos
    """
    term_groups = []
    
    # Define regex pattern to extract FIELD(terms) AND/OR FIELD(terms)
    pattern = r'(TITLE|TITLE-ABS-KEY|AUTHOR|AFFIL|SRCTITLE)\(([^)]+)\)(?:\s+(AND|OR|AND NOT))?'
    
    matches = re.finditer(pattern, query)
    
    for match in matches:
        field = match.group(1)
        terms_str = match.group(2)
        operator = match.group(3) if match.group(3) else "AND"
        
        # Extract individual terms
        terms = []
        in_quotes = False
        current_term = ""
        
        i = 0
        while i < len(terms_str):
            char = terms_str[i]
            
            if char == '"':
                in_quotes = not in_quotes
                current_term += char
            elif char == ' ' and not in_quotes:
                if current_term.lower().strip() not in ('', 'and', 'or'):
                    terms.append(current_term.strip())
                current_term = ""
            elif (char == 'O' and i+1 < len(terms_str) and terms_str[i+1] == 'R' and 
                  (i+2 == len(terms_str) or terms_str[i+2].isspace()) and not in_quotes):
                # Skip "OR" operator
                i += 1  # Skip 'R'
                current_term = ""
            elif (char == 'A' and i+2 < len(terms_str) and terms_str[i+1] == 'N' and 
                  terms_str[i+2] == 'D' and (i+3 == len(terms_str) or terms_str[i+3].isspace()) and 
                  not in_quotes):
                # Skip "AND" operator
                i += 2  # Skip 'ND'
                current_term = ""
            else:
                current_term += char
            
            i += 1
        
        # Add the last term if any
        if current_term.lower().strip() not in ('', 'and', 'or'):
            terms.append(current_term.strip())
        
        # Clean up terms - remove surrounding quotes but keep internal structure
        clean_terms = []
        for term in terms:
            if term.startswith('"') and term.endswith('"'):
                clean_terms.append(term[1:-1])
            else:
                clean_terms.append(term)
                
        if clean_terms:
            term_groups.append({
                'field': field,
                'terms': clean_terms,
                'operator': operator
            })
    
    return term_groups