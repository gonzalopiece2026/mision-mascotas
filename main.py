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

# --- FUNCIÓN AUXILIAR PARA FECHA Y HORA OFICIAL ARGENTINA (UTC-3) ---
def obtener_ahora_argentina():
    return datetime.utcnow() - timedelta(hours=3)

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
tab_alertas, tab_adopciones, tab_reencuentros, tab_guia = st.tabs([
    "🚨 Pérdidas y Hallazgos", 
    "🏡 Perros en Adopción", 
    "✨ Dar de Baja / Reencuentros",
    "📖 ¿Cómo usar la app?"
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
        fecha_desde_filtro = st.date_input("📅 Mostrar alertas OCURRIDAS desde el día:", value=obtener_ahora_argentina().date() - timedelta(days=30), key="mural_filtro_fecha")

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
            
            # Recorremos la lista invertida para mostrar siempre lo más nuevo primero sin recortar datos
            for mascota in reversed(alertas_filtradas_zona):
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
                        url_wa = f"https://wa.me{num_wa}?text={texto_wa}"
                        
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
                st.info("📭 No hay alertas en la base de datos para comparar.")
            else:
                # Procesar la imagen subida para la búsqueda
                file_bytes = np.asarray(bytearray(img_buscar_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                huella_buscada = extraer_huella_segura(img_bgr)
                
                if huella_buscada is None:
                    st.error("⚠️ No se pudo procesar la imagen de búsqueda. Intentá con otra.")
                else:
                    resultados_compatibles = []
                    
                    for mascota in bd_actual:
                        # Filtro opcional por Sexo
                        if sexo_busqueda_input != "Cualquiera / No sé":
                            if mascota.get("sexo", "").lower() != sexo_busqueda_input.lower():
                                continue
                        
                        # Filtro opcional por Zona (Búsqueda cruzada flexible)
                        zona_mascota = mascota.get("zona", "").lower()
                        zona_buscada = zona_busqueda_input.strip().lower()
                        if zona_buscada and (zona_buscada not in zona_mascota and zona_mascota not in zona_buscada):
                            continue
                            
                        # Comparación de huella visual usando Similitud Coseno
                        huella_mascota = mascota.get("huella_visual")
                        if huella_mascota:
                            similitud = cosine_similarity([huella_buscada], [huella_mascota])[0][0]
