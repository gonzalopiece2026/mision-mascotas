import os
import cv2
import json
import asyncio
import urllib.parse
import numpy as np
import streamlit as st
import requests
import cloudinary
import cloudinary.uploader
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURACIÓN DE CLOUDINARY REAL (DATOS DE TU CAPTURA) ---
cloudinary.config(
    cloud_name="rur0qbqsy",
    api_key="241877892786994",
    api_secret="7lE9S4b4N4q-7iA_VfOOfDk4m0g"
)

# --- CONFIGURACIÓN DE LA PÁGINA WEB ---
st.set_page_config(page_title="Misión Mascotas - Red Nacional con IA", page_icon="🐶", layout="centered")

st.title("🐶 Misión Mascotas")
st.write("Plataforma Federal Autónoma: Buscador inteligente por reconocimiento visual con IA para todo el país.")

# Cargamos la IA y la base de datos
@st.cache_resource
def cargar_modelo():
    return YOLO("yolov8n.pt")

modelo = cargar_modelo()
ARCHIVO_BD = "base_datos_mascotas.json"
CARPETA_IMAGENES = "data_perdidos_encontrados_imagenes"

def cargar_base_datos():
    if os.path.exists(ARCHIVO_BD):
        with open(ARCHIVO_BD, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def guardar_base_datos(datos):
    with open(ARCHIVO_BD, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=4)

def extraer_huella(img_bgr):
    try:
        resultados = modelo(img_bgr, verbose=False)
        for resultado in resultados:
            for box in resultado.boxes:
                if int(box.cls) == 16:  # 16 = Perro
                    coordenadas = box.xyxy.tolist()
                    if coordenadas and len(coordenadas) > 0:
                        x1, y1, x2, y2 = map(int, coordenadas)
                        perro_recortado = img_bgr[y1:y2, x1:x2]
                        if perro_recortado.size > 0:
                            img_redim = cv2.resize(perro_recortado, (64, 64))
                            gris = cv2.cvtColor(img_redim, cv2.COLOR_BGR2GRAY)
                            return (gris.flatten() / 255.0).tolist()
    except Exception as e:
        print(f"Error interno al extraer huella: {e}")
    return None

# --- ROBOT EVOLUCIONADO ANTIBLOQUEO ---
async def ejecutar_robot_global(palabra_clave):
    os.makedirs(CARPETA_IMAGENES, exist_ok=True)
    query_busqueda = f"{palabra_clave} site:facebook.com"
    texto_seguro = urllib.parse.quote(query_busqueda)
    link_global = f"https://google.com{texto_seguro}&tbm=isch"
    contador = 0
    headers_simulados = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'es-ES,es;q=0.9',
        'Referer': 'https://google.com'
    }
    try:
        session = requests.Session()
        response = session.get(link_global, headers=headers_simulados, timeout=10)
        if response.status_code != 200:
            return -1
        html = response.text
        pos = 0
        while True:
            pos = html.find('src="https://encrypted', pos)
            if pos == -1 or contador >= 5:
                break
            start = html.find('https://', pos)
            end = html.find('"', start)
            src = html[start:end]
            if src:
                try:
                    contador += 1
                    nombre_archivo = f"robot_global_{contador}.jpg"
                    ruta_completa = os.path.join(CARPETA_IMAGENES, nombre_archivo)
                    img_data = session.get(src, headers=headers_simulados, timeout=5).content
                    with open(ruta_completa, "wb") as f:
                        f.write(img_data)
                except:
                    contador -= 1
            pos = end
        return contador
    except:
        return -1

# --- INTERFAZ GRÁFICA CON NUEVA PESTAÑA DE GALERÍA ---
pestaña_buscar, pestaña_registrar, pestaña_galeria, pestaña_robot, pestaña_donar = st.tabs([
    "🔎 BUSCAR", 
    "📝 ALERTA", 
    "🖼️ GALERÍA",
    "🤖 ROBOT",
    "💝 DONAR"
])

with pestaña_galeria:
    st.subheader("🖼️ Galería Nacional de Mascotas Alertas")
    st.write("Acá aparecen las fotos de los perritos perdidos y encontrados cargados en tiempo real por los vecinos de todo el país.")
    st.divider()
    
    bd = cargar_base_datos()
    if not bd:
        st.info("📭 No hay alertas registradas en este momento. ¡Las nuevas aparecerán acá!")
    else:
        for mascota in reversed(bd):
            col_img, col_info = st.columns([1, 2])
            with col_img:
                if "http" in mascota["ruta_imagen"]:
                    st.image(mascota["ruta_imagen"], width=150)
                else:
                    if os.path.exists(mascota["ruta_imagen"]):
                        st.image(mascota["ruta_imagen"], width=150)
                    else:
                        st.text("📷 Foto no disponible")
            with col_info:
                st.markdown(f"**👤 Responsable:** {mascota['nombre_dueño']}")
                st.markdown(f"**📍 Ubicación:** {mascota['zona']}")
                
                msg_galeria = urllib.parse.quote(f"¡Hola {mascota['nombre_dueño']}! Vi la foto de la mascota que publicaste en la sección de alertas de Misión Mascotas. ¿Sigue estando activa la búsqueda?")
                url_galeria = f"https://wa.me{mascota['contacto']}?text={msg_galeria}"
                st.link_button(f"💬 Hablar con {mascota['nombre_dueño']}", url_galeria)
            st.divider()

with pestaña_donar:
    st.subheader("💝 Apoyá a Misión Mascotas")
    st.write("Esta plataforma es 100% gratuita y libre de publicidad para ayudar a que más familias vuelvan a encontrarse.")
    st.write("Tu donación nos ayuda directamente a mantener los servidores online las 24 horas y seguir mejorando la Inteligencia Artificial.")
    st.divider()
    st.markdown("### 🚀 Mercado Pago (Monto Libre)")
    st.write("Hacé clic en el botón de abajo para colaborar con el monto que vos elijas de forma segura:")
    st.write("") 
    st.link_button("✨ COLABORAR CON MONTO LIBRE", "https://mercadopago.com.ar")

with pestaña_robot:
    st.subheader("Configuración del Robot Rastreador")
    st.write("Escribí qué querés que el robot busque en internet (ej: perro perdido Moreno, perrito extraviado Avellaneda).")
    termino_busqueda = st.text_input("Palabras clave de búsqueda:", value="perro perdido Moreno")
    if st.button("🚀 INICIAR RASTREO INTELIGENTE"):
        with st.spinner("El robot está buscando imágenes públicas en la red... Esperá unos segundos."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            fotos_bajadas = loop.run_until_complete(ejecutar_robot_global(termino_busqueda))
        if fotos_bajadas == -1:
            st.error("❌ Ocurrió un problema de red al procesar la solicitud.")
        elif fotos_bajadas == 0:
            st.warning("⚠️ No se pudieron extraer imágenes en este intento. Probá afinando o cambiando las palabras clave.")
        else:
            st.success(f"🤖 ¡Robot finalizado con éxito! Se completó el rastreo y se descargaron {fotos_bajadas} fotos nuevas.")

with pestaña_registrar:
    st.subheader("Registrar Alerta de Mascota")
    st.write("Subí la foto y los datos del perrito para que el buscador lo indexe.")
    img_file = st.file_uploader("Subí la foto del perro", type=["jpg", "jpeg", "png", "webp"], key="reg_img")
    nombre = st.text_input("Nombre del Dueño / Rescatista")
    zona = st.text_input("Provincia / Localidad / Barrio (Ej: Moreno, Buenos Aires)")
    contacto = st.text_input("Teléfono de Contacto (Con código de área, ej: 1123456789)")
    link = st.text_input("Link de la Publicación (Opcional)")
    if st.button("Guardar en la Red Nacional"):
        if img_file and nombre and zona and contacto:
            contacto_limpio = "".join(filter(str.isdigit, contacto))
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            
            with st.spinner("La IA está analizando la foto..."):
                huella = extraer_huella(img_bgr)
            
            if huella is None:
                st.error("❌ La IA no detectó ningún perro en la foto. Intentá con otra imagen más clara.")
            else:
                with st.spinner("Subiendo imagen de forma segura a la nube de Cloudinary..."):
                    try:
                        cv2.imwrite("temp_upload.jpg", img_bgr)
                        resultado_upload = cloudinary.uploader.upload("temp_upload.jpg")
                        ruta_foto_cloudinary = resultado_upload["secure_url"]
                        if os.path.exists("temp_upload.jpg"):
                            os.remove("temp_upload.jpg")
                    except Exception as upload_err:
                        st.error(f"Error al blindar en la nube: {upload_err}")
                        ruta_foto_cloudinary = f"fotos_registradas/perro_{nombre}_{contacto_limpio}.jpg"
                        cv2.imwrite(ruta_foto_cloudinary, img_bgr)
                
                bd = cargar_base_datos()
                bd.append({
                    "nombre_dueño": nombre,
                    "zona": zona,
                    "contacto": contacto_limpio,
                    "link_redes": link if link else "No especificado",
                    "ruta_imagen": ruta_foto_cloudinary,
                    "huella": huella
                })
                guardar_base_datos(bd)
                st.success(f"✅ ¡Éxito! Mascota registrada de forma permanente en la nube nacional.")
        else:
            st.warning("⚠️ Todos los campos principales son obligatorios.")

with pestaña_buscar:
    st.subheader("Buscar Coincidencias Visuales")
    st.write("Subí la foto de un perro para contrastarlo con la base de datos.")
