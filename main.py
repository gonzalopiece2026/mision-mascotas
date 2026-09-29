import os
import cv2
import json
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

ARCHIVO_BD = "base_datos_mascotas.json"

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
                    porcentaje = float(similitud[0][0]) * 100
                    
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
                    
                    # LINK DE COPIA TRADICIONAL COMPATIBLE CON TODOS LOS SERVIDORES
                    mensaje_whatsapp = urllib.parse.quote(f"¡Hola {responsable_match}! Vi tu alerta en Misión Mascotas publicada el {fecha_hecho_match} en {zona_match}. Encontré una coincidencia visual muy alta con tu perrito. ¿Podemos hablar?")
                    link_final_wa = f"https://wa.me{numero_match}?text={mensaje_whatsapp}"
                    
                    st.subheader("📱 Datos de Contacto Directo")
                    st.code(f"Número del dueño: +{numero_match}", language="text")
                    
                    # Cuadro de texto nativo infalible que permite copiar el link sin trabar la app
                    st.text_input("🔗 Enlace directo de chat (Copialo y pegalo en tu navegador):", value=link_final_wa)
                    st.write("") 
                    
                    # DESPLIEGUE DIRECTO NATIVO DE LA IMAGEN DE CLOUDINARY
                    url_foto_match = mejor_coincidencia.get('ruta_imagen', '')
                    if "http" in url_foto_match:
                        st.image(url_foto_match, caption="Foto oficial de la mascota cargada en el reporte nacional", use_container_width=True)
                    else:
                        st.warning("📷 La foto de este registro viejo no está disponible en la nube de internet.")
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
            with st.spinner("Subiendo imagen de forma segura a la nube de Cloudinary..."):
                try:
                    cv2.imwrite("temp_upload.jpg", img_bgr)
                    resultado_upload = cloudinary.uploader.upload("temp_upload.jpg")
                    ruta_foto_cloudinary = resultado_upload["secure_url"]
                    if os.path.exists("temp_upload.jpg"):
                        os.remove("temp_upload.jpg")
                except Exception as upload_err:
                    print(f"Error al subir: {upload_err}")
                    ruta_foto_cloudinary = "error"
            
            fecha_hecho_str = fecha_suceso.strftime("%d/%m/%Y")
            
            # Sincronización de hora argentina oficial (-3 horas del servidor)
            hora_argentina = datetime.now() - timedelta(hours=3)
            fecha_subida_str = hora_argentina.strftime("%d/%m/%Y a las %H:%M hs")
            
            nueva_mascota = {
                "nombre_dueño": nombre_rescatista,
                "zona": lugar_hecho,
                "contacto": num_limpio,
                "ruta_imagen": ruta_foto_cloudinary,
                "huella": huella,
                "fecha_hecho": fecha_hecho_str,
                "fecha_subida": fecha_subida_str
            }
            
            bd = cargar_base_datos()
            bd.append(nueva_mascota)
            guardar_base_datos(bd)
            st.success("✅ ¡Éxito! Mascota registrada cronológicamente en la nube nacional.")
            st.rerun()
    else:
        st.warning("⚠️ Todos los campos principales son obligatorios.")

st.divider()

# --- 3️⃣ SECCIÓN DE GALERÍA NACIONAL ---
st.header("🖼️ Galería Nacional de Mascotas Alertas")
bd = cargar_base_datos()
if not bd:
    st.info("📭 No hay alertas registradas en este momento. Las nuevas aparecerán acá.")
else:
    for mascara in reversed(bd):
        col_img, col_info = st.columns(2)
        with col_img:
            if "http" in mascara.get("ruta_imagen", ""):
                st.image(mascara["ruta_imagen"], width=150)
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
