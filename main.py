import os
import cv2
import json
import base64
import urllib.parse
from datetime import datetime, timedelta
import numpy as np
import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURACIÓN DE LA PÁGINA WEB ---
st.set_page_config(page_title="Misión Mascotas - Red Nacional con IA", page_icon="🐶", layout="centered")

st.title("🐶 Misión Mascotas")
st.write("Plataforma Federal Autónoma: Buscador inteligente por reconocimiento visual con IA y Red Comunitaria.")

ARCHIVO_BD = "base_datos_mascotas.json"
ARCHIVO_HISTORIAL = "base_datos_reencuentros.json"
ARCHIVO_ADOPCIONES = "base_datos_adopciones.json"

def cargar_base_datos(archivo):
    if os.path.exists(archivo):
        with open(archivo, "r", encoding="utf-8") as f:
            try:
                contenido = f.read().strip()
                if not contenido or contenido == "[]":
                    return []
                f.seek(0)
                return json.load(f)
            except:
                return []
    return []

def guardar_base_datos(datos, archivo):
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=4)

# MOTOR DE EXTRACCIÓN VISUAL EN MATRIZ NATIVA (INFALIBLE)
def extraer_huella_segura(img_bgr):
    try:
        if img_bgr is not None and img_bgr.size > 0:
            img_redim = cv2.resize(img_bgr, (64, 64))
            gris = cv2.cvtColor(img_redim, cv2.COLOR_BGR2GRAY)
            return (gris.flatten() / 255.0).tolist()
    except Exception as e:
        print(f"Error interno en matriz visual: {e}")
    return None

# Carga de bases de datos
bd_actual = cargar_base_datos(ARCHIVO_BD)
bd_reencuentros = cargar_base_datos(ARCHIVO_HISTORIAL)
bd_adopciones = cargar_base_datos(ARCHIVO_ADOPCIONES)

# --- 📊 SECCIÓN DE CONTADORES Y MÉTRICAS EN TIEMPO REAL ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🚨 Alertas Activas", value=len(bd_actual))
with col2:
    st.metric(label="🏡 En Adopción", value=len(bd_adopciones))
with col3:
    st.metric(label="❤️ Reencuentros", value=len(bd_reencuentros))

st.divider()

# --- 💡 CONSEJOS ÚTILES PARA LA COMUNIDAD ---
with st.expander("💡 Consejos útiles para la comunidad"):
    st.markdown("""
    * 💧 **Si encontraste un perro:** Dejale agua fresca, resguardalo en un lugar seguro y sacale una foto clara de frente para subirla a la red.
    * 🏥 **Si se te perdió:** Avisá a veterinarias y refugios de tu zona, pegá carteles y mantené activa la alerta acá en **Misión Mascotas**.
    * 🏡 **Si vas a adoptar:** Recordá que la adopción es un compromiso responsable para toda la vida del animal.
    """)

st.divider()

# --- 📌 NAVEGACIÓN PRINCIPAL POR PESTAÑAS (TABS) ---
tab_alertas, tab_adopciones, tab_reencuentros = st.tabs([
    "🚨 Pérdidas y Hallazgos", 
    "🏡 Perros en Adopción", 
    "✨ Dar de Baja / Reencuentros"
])

# =============================================================================
# 1️⃣ PESTAÑA: PÉRDIDAS Y HALLAZGOS (BUSCADOR + REGISTRO + MURAL)
# =============================================================================
with tab_alertas:
    st.header("📢 Alertas Activas en tu Zona")
    st.write("Visualizá los perros reportados en tu localidad filtrando por zona y fecha del suceso.")

    col_filtro1, col_filtro2 = st.columns(2)
    with col_filtro1:
        zona_filtro_inicio = st.text_input("📍 Filtrar por Zona / Partido / Provincia", value="Moreno", key="mural_filtro_zona")
    with col_filtro2:
        fecha_desde_filtro = st.date_input("📅 Mostrar alertas OCURRIDAS desde el día:", value=datetime.now() - timedelta(days=30), key="mural_filtro_fecha")

    if not bd_actual:
        st.info("📭 No hay alertas activas registradas en el sistema.")
    else:
        zona_buscada_limpia = zona_filtro_inicio.strip().lower()
        fecha_filtro_dt = fecha_desde_filtro
        
        alertas_filtradas_zona = []
        for m in bd_actual:
            zona_mascota = m.get("zona", "").lower()
            fecha_str = m.get("fecha_hecho", "")
            
            cumple_fecha = True
            if fecha_str:
                try:
                    fecha_mascota_dt = datetime.strptime(fecha_str, "%d/%m/%Y").date()
                    if fecha_mascota_dt < fecha_filtro_dt:
                        cumple_fecha = False
                except:
                    pass

            cumple_zona = not zona_buscada_limpia or zona_buscada_limpia in zona_mascota or zona_mascota in zona_buscada_limpia
            
            if cumple_zona and cumple_fecha:
                alertas_filtradas_zona.append(m)
                
        if not alertas_filtradas_zona:
            st.warning(f"⚠️ No hay alertas activas registradas para '{zona_filtro_inicio}' desde el {fecha_desde_filtro.strftime('%d/%m/%Y')}.")
        else:
            st.success(f"📍 Mostrando {len(alertas_filtradas_zona)} alerta(s) activa(s) en **{zona_filtro_inicio}**")
            
            for mascota in reversed(alertas_filtradas_zona[-4:]):
                with st.container(border=True):
                    tipo_alerta_mural = mascota.get('tipo_alerta', 'Perdido')
                    badge_mural = "🔴 PERDIDO" if tipo_alerta_mural == "Perdido" else "🟢 ENCONTRADO"
                    nombre_mural = mascota.get('nombre_perro', 'Sin nombre')
                    
                    st.markdown(f"### {badge_mural} - 🐾 {nombre_mural} ({mascota.get('sexo', '')})")
                    st.info(f"📍 **Zona:** {mascota.get('zona', 'No especificada')} \n📅 **Ocurrió el:** {mascota.get('fecha_hecho', 'No especificada')} \n🕒 **Publicado:** {mascota.get('fecha_subida', 'Recientemente')} \n🐕 **Raza / Color:** {mascota.get('raza', '')} | {mascota.get('color', '')}")
                    
                    numero_mural = mascota.get('contacto', '')
                    if numero_mural:
                        num_wa = "".join(filter(str.isdigit, numero_mural))
                        if num_wa and not num_wa.startswith("54"):
                            num_wa = "549" + num_wa
                        
                        texto_wa = urllib.parse.quote(f"Hola, vi tu reporte en Misión Mascotas sobre {nombre_mural}. ¡Quería comunicarme con vos!")
                        url_wa = f"https://wa.me/{num_wa}?text={texto_wa}"
                        
                        st.markdown(f'''
                            <a href="{url_wa}" target="_blank" style="text-decoration: none;">
                                <div style="background-color: #25D366; color: white; padding: 10px 15px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 15px; margin-top: 5px; margin-bottom: 10px;">
                                    💬 Contactar por WhatsApp ({numero_mural})
                                </div>
                            </a>
                        ''', unsafe_allow_html=True)
                    
                    foto_b64 = mascota.get('ruta_imagen', '')
                    if foto_b64 and foto_b64 != "error":
                        try:
                            bytes_decor = base64.b64decode(foto_b64)
                            st.image(bytes_decor, use_container_width=True)
                        except:
                            pass

    st.divider()

    # --- BÚSQUEDA INTELIGENTE IA ---
    st.header("🔎 Buscar Coincidencias Inteligentes con IA")
    st.write("Subí la foto para buscar cruce por reconocimiento visual.")
    img_buscar_file = st.file_uploader("Subí la foto para buscar", type=["jpg", "jpeg", "png", "webp"], key="bus_img")
    zona_busqueda_input = st.text_input("Zona, Barrio o Provincia donde buscás", key="bus_zona")
    sexo_busqueda_input = st.selectbox("Sexo del perro que buscás", ["Cualquiera / No sé", "Macho", "Hembra"], key="bus_sexo")

    if st.button("Buscar Coincidencias con IA", key="btn_buscar_principal"):
        if img_buscar_file:
            if not bd_actual:
                st.warning("📭 La base de datos está vacía en este momento.")
            else:
                file_bytes = np.asarray(bytearray(img_buscar_file.read()), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, 1)
                huella_usuario = extraer_huella_segura(img_bgr)
                
                if huella_usuario is None:
                    st.error("❌ No se pudo procesar la imagen de búsqueda.")
                else:
                    coincidencias_encontradas = []
                    vector_u = np.array(huella_usuario).reshape(1, -1)
                    zona_usuario = zona_busqueda_input.strip().lower()
                    
                    for mascota in bd_actual:
                        huella_db = np.array(mascota.get("huella", [])).reshape(1, -1)
                        similitud = cosine_similarity(vector_u, huella_db)
                        porcentaje = float(np.squeeze(similitud)) * 100
                        
                        if porcentaje >= 65:
                            mascota_con_score = mascota.copy()
                            mascota_con_score["porcentaje_match"] = porcentaje
                            coincidencias_encontradas.append(mascota_con_score)
                    
                    coincidencias_encontradas = sorted(coincidencias_encontradas, key=lambda x: x["porcentaje_match"], reverse=True)
                    
                    if not coincidencias_encontradas:
                        st.error("❌ No se encontraron coincidencias similares.")
                    else:
                        st.success(f"📊 ¡Se encontraron {len(coincidencias_encontradas)} posibles coincidencias!")
                        for idx, coincidencia in enumerate(coincidencias_encontradas):
                            with st.container(border=True):
                                st.markdown(f"### 🎯 Opción #{idx + 1} - Similitud Visual: {coincidencia['porcentaje_match']:.2f}%")
                                st.info(f"👤 **Responsable:** {coincidencia.get('nombre_dueño', 'Anónimo')} \n📍 **Lugar:** {coincidencia.get('zona', 'No especificado')} \n🐾 **Nombre:** {coincidencia.get('nombre_perro', 'No especificado')}")
                                
                                foto_b64 = coincidencia.get('ruta_imagen', '')
                                if foto_b64 and foto_b64 != "error":
                                    try:
                                        bytes_decor = base64.b64decode(foto_b64)
                                        st.image(bytes_decor, use_container_width=True)
                                    except:
                                        pass

    st.divider()

    # --- REGISTRO DE ALERTA ---
    st.header("📝 Registrar Alerta de Mascota Perdida / Encontrada")
    tipo_alerta = st.selectbox("¿Qué tipo de alerta querés crear?", ["Perdido", "Encontrado"])
    img_file = st.file_uploader("Subí la foto de la mascota", type=["jpg", "jpeg", "png", "webp"], key="reg_img")
    nombre_perro = st.text_input("Nombre de la mascota")
    sexo_perro = st.selectbox("Sexo de la mascota", ["Macho", "Hembra", "No especificado"], key="reg_sexo")
    raza_perro = st.text_input("Raza (Ej: Cruza, Caniche)")
    color_perro = st.text_input("Color del pelaje")
    detalles_perro = st.text_area("Detalles particulares")
    nombre_rescatista = st.text_input("Nombre del Dueño / Rescatista")
    lugar_hecho = st.text_input("¿Dónde ocurrió? (Ej: Moreno)", key="casillero_lugar_hecho_zona")
    fecha_suceso = st.date_input("¿Qué día ocurrió?", value=datetime.now())
    telefono_contacto = st.text_input("Teléfono de Contacto (Ej: 1162330944)", key="casillero_registro_telefono")

    if st.button("Guardar Alerta en la Red", key="btn_guardar_principal"):
        num_limpio = "".join(filter(str.isdigit, telefono_contacto))
        if not img_file or not nombre_rescatista or not lugar_hecho or not telefono_contacto:
            st.warning("⚠ Todos los campos principales son obligatorios.")
        elif len(num_limpio) < 10:
            st.error("❌ El teléfono debe tener al menos 10 dígitos.")
        else:
            bytes_datos_foto = img_file.getvalue()
            file_bytes = np.asarray(bytearray(bytes_datos_foto), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            huella = extraer_huella_segura(img_bgr)
            
            foto_b64_string = base64.b64encode(bytes_datos_foto).decode('utf-8')
            bd = cargar_base_datos(ARCHIVO_BD)
            bd.append({
                "tipo_alerta": str(tipo_alerta),
                "nombre_perro": str(nombre_perro).strip() if nombre_perro else "No especificado",
                "sexo": str(sexo_perro),
                "raza": str(raza_perro) if raza_perro else "No específica",
                "color": str(color_perro) if color_perro else "No especificado",
                "detalles": str(detalles_perro) if detalles_perro else "Sin detalles",
                "nombre_dueño": str(nombre_rescatista),
                "zona": str(lugar_hecho),
                "contacto": str(num_limpio),
                "ruta_imagen": str(foto_b64_string),
                "huella": huella,
                "fecha_hecho": fecha_suceso.strftime("%d/%m/%Y"),
                "fecha_subida": (datetime.now() - timedelta(hours=3)).strftime("%d/%m/%Y a las %H:%M hs")
            })
            guardar_base_datos(bd, ARCHIVO_BD)
            st.success("✅ Alerta guardada con éxito.")
            st.rerun()


# =============================================================================
# 2️⃣ PESTAÑA: ADOPCIONES RESPONSABLES
# =============================================================================
with tab_adopciones:
    st.header("🏡 Perros en Adopción Responsable")
    st.write("Conocé a los perritos que están buscando un hogar definitivo.")

    # MURAL DE ADOPCIONES
    if not bd_adopciones:
        st.info("📌 Por el momento no hay perritos registrados en adopción. Podés publicar uno abajo.")
    else:
        for adp in reversed(bd_adopciones):
            with st.container(border=True):
                st.markdown(f"### 🐾 {adp.get('nombre', 'Sin nombre')} ({adp.get('sexo', '')}) - {adp.get('edad', 'Edad no especificada')}")
                st.info(f"📍 **Zona:** {adp.get('zona', 'No especificada')} \n🐕 **Tamaño / Raza:** {adp.get('tamano', '')} | {adp.get('raza', '')} \n📝 **Historia / Salud:** {adp.get('detalles', '')}")
                
                num_adp = adp.get('contacto', '')
                if num_adp:
                    num_wa = "".join(filter(str.isdigit, num_adp))
                    if num_wa and not num_wa.startswith("54"):
                        num_wa = "549" + num_wa
                    
                    texto_wa = urllib.parse.quote(f"Hola, vi en Misión Mascotas que {adp.get('nombre', 'el perrito')} está en adopción. ¡Quería coordinar para conocerlo/a!")
                    url_wa = f"https://wa.me/{num_wa}?text={texto_wa}"
                    
                    st.markdown(f'''
                        <a href="{url_wa}" target="_blank" style="text-decoration: none;">
                            <div style="background-color: #25D366; color: white; padding: 10px 15px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 15px; margin-top: 5px; margin-bottom: 10px;">
                                💛 Consultar para Adoptar por WhatsApp ({num_adp})
                            </div>
                        </a>
                    ''', unsafe_allow_html=True)
                
                foto_b64 = adp.get('ruta_imagen', '')
                if foto_b64 and foto_b64 != "error":
                    try:
                        bytes_decor = base64.b64decode(foto_b64)
                        st.image(bytes_decor, use_container_width=True)
                    except:
                        pass

    st.divider()

    # FORMULARIO PARA PUBLICAR ADOPCIÓN
    st.header("📝 Publicar Mascota en Adopción")
    img_adp = st.file_uploader("Subí la foto del perrito en adopción", type=["jpg", "jpeg", "png", "webp"], key="adp_img")
    nom_adp = st.text_input("Nombre del perro", key="adp_nom")
    sex_adp = st.selectbox("Sexo", ["Macho", "Hembra"], key="adp_sex")
    edad_adp = st.text_input("Edad aproximada (Ej: 3 meses, 2 años)", key="adp_edad")
    tam_adp = st.selectbox("Tamaño estimado", ["Cachorro", "Pequeño", "Mediano", "Grande"], key="adp_tam")
    det_adp = st.text_area("Historia / Vacunas / Castración / Carácter", key="adp_det")
    zona_adp = st.text_input("Zona / Localidad (Ej: Moreno)", key="adp_zona")
    tel_adp = st.text_input("Teléfono de contacto del Rescatista/Tránsito", key="adp_tel")

    if st.button("Publicar en Adopción", key="btn_guardar_adopcion"):
        num_limpio = "".join(filter(str.isdigit, tel_adp))
        if not img_adp or not nom_adp or not zona_adp or not tel_adp:
            st.warning("⚠ Completá la foto, nombre, zona y teléfono de contacto.")
        elif len(num_limpio) < 10:
            st.error("❌ El teléfono debe tener al menos 10 dígitos.")
        else:
            bytes_foto = img_adp.getvalue()
            foto_b64 = base64.b64encode(bytes_foto).decode('utf-8')
            
            bd_adp = cargar_base_datos(ARCHIVO_ADOPCIONES)
            bd_adp.append({
                "nombre": str(nom_adp).strip(),
                "sexo": str(sex_adp),
                "edad": str(edad_adp) if edad_adp else "No especificada",
                "tamano": str(tam_adp),
                "detalles": str(det_adp) if det_adp else "Sin detalles",
                "zona": str(zona_adp),
                "contacto": str(num_limpio),
                "ruta_imagen": str(foto_b64),
                "fecha_publicacion": (datetime.now() - timedelta(hours=3)).strftime("%d/%m/%Y")
            })
            guardar_base_datos(bd_adp, ARCHIVO_ADOPCIONES)
            st.success("✅ ¡Publicación de adopción creada con éxito!")
            st.rerun()


# =============================================================================
# 3️⃣ PESTAÑA: DAR DE BAJA / REENCUENTROS
# =============================================================================
with tab_reencuentros:
    st.header("✨ Misión Cumplida: Dar de Baja Alerta")
    st.write("Ingresá el número con el que publicaste el reporte para moverlo al muro de historias felices.")

    baja_telefono = st.text_input("Número de teléfono (Ej: 1162330944)", key="casillero_baja")

    if st.button("Eliminar y Registrar Reencuentro", key="btn_baja_principal"):
        num_baja_limpio = "".join(filter(str.isdigit, baja_telefono))
        
        if not num_baja_limpio:
            st.warning("⚠️ Ingresá un número válido.")
        else:
            bd = cargar_base_datos(ARCHIVO_BD)
            if not bd:
                st.info("ℹ️ La base de datos está vacía.")
            else:
                mascotas_a_borrar = [m for m in bd if m.get("contacto") == num_baja_limpio]
                nueva_bd = [m for m in bd if m.get("contacto") != num_baja_limpio]
                
                if len(mascotas_a_borrar) > 0:
                    historial = cargar_base_datos(ARCHIVO_HISTORIAL)
                    for mascota in mascotas_a_borrar:
                        mascota["fecha_reencuentro"] = datetime.now().strftime("%d/%m/%Y")
                        historial.append(mascota)
                    
                    guardar_base_datos(nueva_bd, ARCHIVO_BD)
                    guardar_base_datos(historial, ARCHIVO_HISTORIAL)
                    
                    st.success("❤️ ¡Misión cumplida! Se registró el reencuentro con éxito.")
                    st.rerun()
                else:
                    st.error("❌ No se encontró publicación activa con ese número.")

    st.divider()

    st.header("❤️ Muro de Reencuentros Felices")
    if not bd_reencuentros:
        st.info("📌 Todavía no hay reencuentros registrados.")
    else:
        for idx, reencuentro in enumerate(reversed(bd_reencuentros)):
            with st.container(border=True):
                st.markdown(f"### 🎉 ¡Reencuentro Exitoso #{len(bd_reencuentros) - idx}!")
                st.success(f"🐾 **Mascota:** {reencuentro.get('nombre_perro', 'Desconocido')} ({reencuentro.get('sexo', '')}) \n📍 **Zona:** {reencuentro.get('zona', 'No especificada')} \n📅 **Volvió a casa el:** {reencuentro.get('fecha_reencuentro', 'Reciente')}")
                
                foto_b64 = reencuentro.get('ruta_imagen', '')
                if foto_b64 and foto_b64 != "error":
                    try:
                        bytes_decor = base64.b64decode(foto_b64)
                        st.image(bytes_decor, use_container_width=True)
                    except:
                        pass
