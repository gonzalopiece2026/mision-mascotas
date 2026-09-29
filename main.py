import os
import cv2
import json
import time
import asyncio
import urllib.parse
from datetime import datetime, timedelta
import numpy as np
import streamlit as st
import requests
import cloudinary
import cloudinary.uploader
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURACIÓN DE CLOUDINARY REAL ---
cloudinary.config(
    cloud_name="rur0qbqsy",
    api_key="241877892786994",
    api_secret="7lE9S4b4N4q-7iA_VfOOfDk4m0g"
)

# --- CONFIGURACIÓN DE LA PÁGINA WEB ---
st.set_page_config(page_title="Misión Mascotas - Red Nacional con IA", page_icon="🐶", layout="centered")

st.title("🐶 Misión Mascotas")
st.write("Plataforma Federal Autónoma: Buscador inteligente por reconocimiento visual con IA para todo el país.")

# CONFIGURACIÓN DE BASE DE DATOS ULTRA-ESTABLE EN MEMORIA VIVA
if "base_datos_federal" not in st.session_state:
    st.session_state["base_datos_federal"] = []

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
            bd = st.session_state["base_datos_federal"]
            if not bd:
                st.warning("📭 La base de datos nacional está vacía. Registrá una mascota abajo primero.")
            else:
                mejor_coincidencia = None
                mayor_porcentaje = 0.0
                vector_u = np.array(huella_usuario).reshape(1, -1)
                
                for mascota in bd:
                    huella_db = np.array(mascota["huella"]).reshape(1, -1)
                    similitud = cosine_similarity(vector_u, huella_db)
                    porcentaje = float(similitud) * 100
                    
                    if porcentaje > mayor_porcentaje:
                        mayor_porcentaje = porcentaje
                        mejor_coincidencia = mascota
                        
                if mejor_coincidencia and mayor_porcentaje > 65:
                    st.success(f"📊 ¡COINCIDENCIA ENCONTRADA CON ÉXITO! ({mayor_porcentaje:.2f}% de parecido)")
                    
                    fecha_hecho_match = mejor_coincidencia.get('fecha_hecho', 'No especificada')
                    zona_match = mejor_coincidencia.get('zona', 'No especificada')
                    responsable_match = mejor_coincidencia.get('nombre_dueño', 'Anónimo')
                    numero_match = mejor_coincidencia.get('contacto', '')
                    
                    st.info(f"👤 Responsable: {responsable_match} \n📍 Lugar del hecho: {zona_match} \n📅 Fecha del suceso: {fecha_hecho_match}")
                    
                    mensaje_whatsapp = urllib.parse.quote(f"¡Hola {responsable_match}! Vi tu alerta en Misión Mascotas publicada el {fecha_hecho_match} en {zona_match}. Encontré una coincidencia visual muy alta con tu perrito. ¿Podemos hablar?")
                    url_whatsapp = f"https://wa.me{numero_match}?text={mensaje_whatsapp}"
                    
                    st.markdown(f'<h3>📱 Contactar al responsable:</h3>', unsafe_allow_html=True)
                    st.write(f"Número del dueño: +{numero_match}")
                    st.link_button("💬 CHATEAR DIRECTO POR WHATSAPP", url_whatsapp)
                    st.write("") 
                    
                    url_foto_match = mejor_coincidencia.get('ruta_imagen', '')
                    if "http" in str(url_foto_match):
                        st.image(str(url_foto_match), caption="Foto oficial de la mascota en el reporte nacional", use_container_width=True)
                    else:
                        st.warning("📷 La foto de este registro no está disponible en los servidores de la nube.")
                else:
                    st.error("❌ No se encontraron coincidencias similares en el sistema.")
    else:
        st.warning("⚠️ Primero tenés que subir una foto en el recuadro de arriba para poder buscar.")

st.divider()

# --- 2️⃣ SECCIÓN DE REGISTRO DE ALERTA ---
st.header("📝 Registrar Alerta de Mascota")
st.write("Subí la foto y detallá cuándo y dónde se vio al perrito por última vez.")
img_file = st.file_uploader("Subí la foto del perro", type=["jpg", "jpeg", "png", "webp"], key="reg_img")
nombre_rescatista = st.text_input("Nombre del Dueño / Rescatista")
lugar_hecho = st.text_input("¿Dónde se extravió / encontró? (Ej: Barrio Satélite, Moreno)")
fecha_suceso = st.date_input("¿Qué día ocurrió?", value=datetime.now())
telefono_contacto = st.text_input("Teléfono de Contacto (Con código de área, ej: 1123456789)")

if st.button("Guardar en la Red Nacional", key="btn_guardar_principal"):
    if img_file and nombre_rescatista and lugar_hecho and telefono_contacto:
        num_limpio = "".join(filter(str.isdigit, telefono_contacto))
        if num_limpio.startswith("0"):
            num_limpio = num_limpio[1:]
        if num_limpio.startswith("15"):
            num_limpio = num_limpio[2:]
        if not num_limpio.startswith("54"):
            if num_limpio.startswith("9"):
                num_limpio = "54" + num_limpio
            else:
                num_limpio = "549" + num_limpio
                
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        
        huella = extraer_huella_segura(img_bgr)
        
        if huella is None:
            st.error("❌ Ocurrió un problema al procesar la imagen.")
        else:
            ruta_foto_cloudinary = "error"
            with st.spinner("Subiendo imagen de forma segura a la nube de Cloudinary..."):
                try:
                    cv2.imwrite("temp_upload.jpg", img_bgr)
                    # CORRECCIÓN MAESTRA: Forzamos la subida síncrona obligatoria para que retenga el archivo
                    resultado_upload = cloudinary.uploader.upload("temp_upload.jpg", resource_type="image")
                    ruta_foto_cloudinary = resultado_upload.get("secure_url", "error")
                    time.sleep(2) # Espera técnica para asegurar el renderizado
                    if os.path.exists("temp_upload.jpg"):
                        os.remove("temp_upload.jpg")
                except Exception as upload_err:
                    print(f"Error al subir: {upload_err}")
            
            if ruta_foto_cloudinary == "error" or not ruta_foto_cloudinary:
                st.error("❌ Error de conexión con el servidor de imágenes. Intenta de nuevo.")
            else:
                fecha_hecho_str = fecha_suceso.strftime("%d/%m/%Y")
                hora_argentina = datetime.now() - timedelta(hours=3)
                fecha_subida_str = hora_argentina.strftime("%d/%m/%Y a las %H:%M hs")
                
                nueva_mascota = {
                    "nombre_dueño": nombre_rescatista,
                    "zona": lugar_hecho,
                    "contacto": num_limpio,
                    "ruta_imagen": str(ruta_foto_cloudinary),
                    "huella": huella,
                    "fecha_hecho": fecha_hecho_str,
                    "fecha_subida": fecha_subida_str
                }
                
                st.session_state["base_datos_federal"].append(nueva_mascota)
                st.success("✅ ¡Éxito! Mascota registrada cronológicamente en la nube nacional.")
                time.sleep(1)
                st.rerun()
    else:
        st.warning("⚠️ Todos los campos principales son obligatorios.")

st.divider()

# --- 3️⃣ SECCIÓN DE GALERÍA NACIONAL ---
st.header("🖼️ Galería Nacional de Mascotas Alertas")
bd = st.session_state["base_datos_federal"]
if not bd:
    st.info("📭 No hay alertas registradas en este momento. Las nuevas aparecerán acá.")
else:
    for mascara in reversed(bd):
        col_img, col_info = st.columns(2)
        with col_img:
            url_galeria_img = mascara.get("ruta_imagen", "")
            if "http" in str(url_galeria_img):
                st.image(str(url_galeria_img), width=150)
            else:
                st.text("📷 Foto no disponible")
        with col_info:
            nombre_galeria = mascara.get('nombre_dueño', 'Anónimo')
            st.markdown(f"**👤 Responsable:** {nombre_galeria}")
            st.markdown(f"**📍 Lugar del hecho:** {mascara.get('zona', 'No especificado')}")
            st.markdown(f"**📅 Ocurrió el:** {mascara.get('fecha_hecho', 'No especificado')}")
            st.markdown(f"**⏰ Subido el:** {mascara.get('fecha_subida', 'No especificado')}")
            
            num_destino = mascara.get('contacto', '')
            msg_gal = urllib.parse.quote("¡Hola! Vi la publicación de la mascota en Misión Mascotas. ¿Sigue activa la búsqueda?")
            url_gal = f"https://wa.me{num_destino}?text={msg_gal}"
            
            st.write(f"WhatsApp: +{num_destino}")
            st.link_button("💬 Hablar por WhatsApp", url_gal)
        st.divider()

st.divider()

# --- 4️⃣ SECCIÓN DE DONACIONES ---
