import os
import cv2
import json
import urllib.parse
from datetime import datetime, timedelta
import numpy as np
import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURACIÓN DE LA PÁGINA WEB ---
st.set_page_config(page_title="Misión Mascotas - Red Nacional con IA", page_icon="🐶", layout="centered")

st.title("🐶 Misión Mascotas")
st.write("Plataforma Federal Autónoma: Buscador inteligente por reconocimiento visual con IA para todo el país.")

ARCHIVO_BD = "base_datos_mascotas.json"
CARPETA_FOTOS = "fotos_mascotas"

# Creamos la carpeta de almacenamiento visual nativo si no existe
if not os.path.exists(CARPETA_FOTOS):
    os.makedirs(CARPETA_FOTOS)

def cargar_base_datos():
    if os.path.exists(ARCHIVO_BD):
        with open(ARCHIVO_BD, "r", encoding="utf-8") as f:
            try:
                contenido = f.read().strip()
                if not contenido or contenido == "[]":
                    return []
                f.seek(0)
                return json.load(f)
            except:
                return []
    return []

def guardar_base_datos(datos):
    with open(ARCHIVO_BD, "w", encoding="utf-8") as f:
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

# --- 1️⃣ SECCIÓN DE BÚSQUEDA INTELIGENTE ---
st.header("🔎 Buscar Coincidencias Visuales")
st.write("Subí la foto de un perro para contrastarlo de forma instantánea con toda la red nacional.")
img_buscar_file = st.file_uploader("Subí la foto para buscar", type=["jpg", "jpeg", "png", "webp"], key="bus_img")

if st.button("Buscar Coincidencias con IA", key="btn_buscar_principal"):
    if img_buscar_file:
        file_bytes = np.asarray(bytearray(img_buscar_file.read()), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        
        with st.spinner("Buscando coincidencias en la base de datos..."):
            huella_usuario = extraer_huella_segura(img_bgr)
            
        if huella_usuario is None:
            st.error("❌ No se pudo procesar la imagen de búsqueda.")
        else:
            bd = cargar_base_datos()
            if not bd:
                st.warning("📭 La base de datos nacional está vacía. Registrá una mascota abajo primero.")
            else:
                mejor_coincidencia = None
                mayor_porcentaje = 0.0
                vector_u = np.array(huella_usuario).reshape(1, -1)
                
                for mascota in bd:
                    huella_db = np.array(mascota["huella"]).reshape(1, -1)
                    similitud = cosine_similarity(vector_u, huella_db)
                    
                    # BLINDAJE MATEMÁTICO ABSOLUTO: Extraemos la posición exacta para pulverizar el TypeError
                    porcentaje = float(similitud[0][0]) * 100
                    
                    if porcentaje > mayor_porcentaje:
                        mayor_porcentaje = porcentaje
                        mejor_coincidencia = mascota
                        
                if mejor_coincidencia and mayor_porcentaje > 65:
                    st.success(f"📊 ¡COINCIDENCIA ENCONTRADA CON ÉXITO! ({mayor_porcentaje:.2f}% de parecido)")
                    
                    tipo_match = mejor_coincidencia.get('tipo_alerta', 'Perdido')
                    cartel_tipo = "🔴 ALERTA: PERDIDO" if tipo_match == "Perdido" else "🟢 ALERTA: ENCONTRADO"
                    
                    st.markdown(f"### {cartel_tipo}")
                    st.info(f"👤 **Responsable:** {mejor_coincidencia.get('nombre_dueño', 'Anónimo')} \n📍 **Lugar del hecho:** {mejor_coincidencia.get('zona', 'No especificado')} \n📅 **Fecha del suceso:** {mejor_coincidencia.get('fecha_hecho', 'No especificada')} \n🐾 **Nombre de la mascota:** {mejor_coincidencia.get('nombre_perro', 'No especificado')} \n🐕 **Raza / Color:** {mejor_coincidencia.get('raza', 'No especificada')} | {mejor_coincidencia.get('color', 'No especificado')} \n📝 **Detalles particulares:** {mejor_coincidencia.get('detalles', 'Sin detalles adicionales')}")
                    
                    # CONTACTO MANUAL INDESTRUCTIBLE
                    numero_match = mejor_coincidencia.get('contacto', '')
                    st.markdown("### 📱 Teléfono de Contacto:")
                    st.write("Copiá el número de abajo para comunicarte con el responsable:")
                    st.code(f"{numero_match}", language="text")
                    st.write("") 
                    
                    # RENDERIZADO DE IMAGEN LOCAL DIRECTA EN EL MATCH
                    ruta_foto_match = mejor_coincidencia.get('ruta_imagen', '')
                    if ruta_foto_match and os.path.exists(ruta_foto_match):
                        st.image(ruta_foto_match, caption="Foto oficial del cruce inteligente", use_container_width=True)
                    else:
                        st.warning("📷 La foto de este registro no está disponible.")
                else:
                    st.error("❌ No se encontraron coincidencias similares en el sistema.")
    else:
        st.warning("⚠️ Primero tenés que subir una foto en el recuadro de arriba para poder buscar.")

st.divider()

# --- 2️⃣ SECCIÓN DE REGISTRO DE ALERTA ---
st.header("📝 Registrar Alerta de Mascota")
st.write("Subí la foto y detallá las características del animal para agilizar el cruce inteligente.")

# CASILLEROS COMUNITARIOS COMPLETOS FIJOS
tipo_alerta = st.selectbox("¿Qué tipo de alerta querés crear?", ["Perdido", "Encontrado"])
img_file = st.file_uploader("Subí la foto de la mascota", type=["jpg", "jpeg", "png", "webp"], key="reg_img")
nombre_perro = st.text_input("Nombre de la mascota (Si no lo sabés, poné 'No lo sé')")
raza_perro = st.text_input("Raza (Ej: Cruza, Caniche, Ovejero)")
color_perro = st.text_input("Color principal del pelaje")
detalles_perro = st.text_area("Detalles particulares (Ej: Tiene collar rojo, renguea de una pata, es asustadizo)")

st.write("---")
st.write("📋 Datos del Responsable:")
nombre_rescatista = st.text_input("Nombre del Dueño / Rescatista")
lugar_hecho = st.text_input("¿Dónde ocurrió? (Ej: Barrio Satélite, Moreno)")
fecha_suceso = st.date_input("¿Qué día ocurrió?", value=datetime.now())
telefono_contacto = st.text_input("Teléfono de Contacto (Ej: 1162330944)")

if st.button("Guardar en la Red Nacional", key="btn_guardar_principal"):
    if img_file and nombre_rescatista and lugar_hecho and telefono_contacto:
        num_limpio = "".join(filter(str.isdigit, telefono_contacto))
                
        file_bytes = np.asarray(bytearray(img_file.read()), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        
        huella = extraer_huella_segura(img_bgr)
        
        if huella is None:
            st.error("❌ Ocurrió un problema al procesar la imagen.")
        else:
            fecha_hecho_str = fecha_suceso.strftime("%d/%m/%Y")
            hora_argentina = datetime.now() - timedelta(hours=3)
            fecha_subida_str = hora_argentina.strftime("%d/%m/%Y a las %H:%M hs")
            
            # GUARDADO DE IMAGEN NATIVO ULTRA VISIBLE EN SERVIDOR LOCAL
            nombre_archivo_foto = f"foto_{num_limpio}_{int(time.time())}.jpg" if 'time' in globals() else f"foto_{num_limpio}_{num_limpio}.jpg"
            ruta_final_guardado = os.path.join(CARPETA_FOTOS, nombre_archivo_foto)
            cv2.imwrite(ruta_final_guardado, img_bgr)
            
            nueva_mascota = {
                "tipo_alerta": str(tipo_alerta),
                "nombre_perro": str(nombre_perro) if nombre_perro else "No especificado",
                "raza": str(raza_perro) if raza_perro else "No específica",
                "color": str(color_perro) if color_perro else "No especificado",
                "detalles": str(detalles_perro) if detalles_perro else "Sin detalles",
                "nombre_dueño": str(nombre_rescatista),
                "zona": str(lugar_hecho),
                "contacto": str(num_limpio),
                "ruta_imagen": str(ruta_final_guardado), # Guardamos la ruta física directa de la foto
                "huella": huella,
                "fecha_hecho": str(fecha_hecho_str),
                "fecha_subida": str(fecha_subida_str)
            }
            
            bd = cargar_base_datos()
            bd.append(nueva_mascota)
            guardar_base_datos(bd)
            st.success("✅ ¡Éxito! Mascota registrada cronológicamente en la red nacional.")
            st.rerun()
    else:
        st.warning("⚠️ Todos los campos principales son obligatorios (Foto, Responsable, Lugar y Teléfono).")

st.divider()

# --- 3️⃣ SECCIÓN DE GALERÍA NACIONAL (FORMATO PIZARRA VERTICAL TIPO TELEVISIÓN) ---
st.header("🖼️ Galería Nacional de Mascotas Alertas")
bd = cargar_base_datos()
if not bd:
    st.info("📭 No hay alertas registradas en este momento. Las nuevas aparecerán acá.")
else:
    for mascara in reversed(bd):
        # Cada registro se clava en un contenedor gris fijo indestructible tipo noticiero
        with st.container(border=True):
            t_alerta = mascara.get('tipo_alerta', 'Perdido')
            cartel_galeria = "🔴 MASCOTA PERDIDA" if t_alerta == "Perdido" else "🟢 MASCOTA ENCONTRADA"
            st.markdown(f"## {cartel_galeria}")
            
            # Ficha técnica fija obligatoria en pantalla
            st.markdown(f"**🐾 Nombre de la mascota:** {mascara.get('nombre_perro', 'No especificado')}")
            st.markdown(f"**🐕 Raza / Color:** {mascara.get('raza', 'No especificada')} | {mascara.get('color', 'No especificado')}")
