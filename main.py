import os
import cv2
import json
import asyncio
import urllib.parse
import numpy as np
import streamlit as st
import requests
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURACIÓN DE LA PÁGINA WEB ---
st.set_page_config(page_title="Misión Mascotas - Red Nacional con IA", page_icon="🐶", layout="centered")

# --- DISEÑO VISUAL PINTORESCO PERSONALIZADO (CSS) ---
st.markdown("""
    <style>
    /* Fondo principal y textos */
    .stApp {
        background-color: #f4f6f9;
    }
    h1 {
        color: #1e3a8a !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px !important;
    }
    .subtitulo {
        text-align: center;
        color: #4b5563;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    /* Estilo para ajustar las pestañas y que no se corten */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
        display: flex;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 8px 16px;
        border: 1px solid #e5e7eb;
        font-weight: 600;
        color: #4b5563;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
        border: 1px solid #2563eb;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
    }
    /* Tarjetas de perros encontrados */
    .tarjeta-perro {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #10b981;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    /* Botón de WhatsApp */
    .btn-whatsapp {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: #25d366;
        color: white !important;
        padding: 12px 24px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        font-size: 1rem;
        box-shadow: 0 4px 6px rgba(37, 211, 102, 0.3);
        transition: background-color 0.3s;
        margin-top: 10px;
    }
    .btn-whatsapp:hover {
        background-color: #128c7e;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🐶 Misión Mascotas</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitulo'>Plataforma Federal Autónoma: Buscador inteligente por reconocimiento visual con IA.</p>", unsafe_allow_html=True)

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
    resultados = modelo(img_bgr, verbose=False)
    for resultado in resultados:
        for box in resultado.boxes:
            if int(box.cls) == 16:  # 16 = Perro
                coordenadas = box.xyxy.tolist()
                x1, y1, x2, y2 = map(int, coordenadas)
                perro_recortado = img_bgr[y1:y2, x1:x2]
                img_redim = cv2.resize(perro_recortado, (64, 64))
                gris = cv2.cvtColor(img_redim, cv2.COLOR_BGR2GRAY)
                return (gris.flatten() / 255.0).tolist()
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

# --- INTERFAZ GRÁFICA CORREGIDA ---
pestaña_buscar, pestaña_registrar, pestaña_robot = st.tabs([
    "🔎 BUSCAR COINCIDENCIA", 
    "📝 CREAR ALERTA", 
    "🤖 ROBOT RASTREADOR"
])

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
            st.success(f"🤖 ¡Robot finalizado con éxito! Se completó el rastreo y se descargaron {fotos_bajadas} fotos nuevas en la base de datos temporal.")

with pestaña_registrar:
    st.subheader("Registrar Alerta de Mascota")
    st.write("Subí la foto y los datos del perrito para que el buscador federal lo indexe.")
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
            huella = extraer_huella(img_bgr)
            
            if huella is None:
                st.error("❌ La IA no detectó ningún perro en la foto. Intentá con otra imagen más clara.")
            else:
                os.makedirs("fotos_registradas", exist_ok=True)
                ruta_foto = f"fotos_registradas/perro_{nombre}_{contacto_limpio}.jpg"
                cv2.imwrite(ruta_foto, img_bgr)
                
                bd = cargar_base_datos()
                bd.append({
                    "nombre_dueño": nombre, "zona": zona, "contacto": contacto_limpio,
                    "link_redes": link if link else "No especificado",
                    "ruta_imagen": ruta_foto, "huella": huella
                })
                guardar_base_datos(bd)
                st.success(f"✅ ¡Éxito! Mascota de '{nombre}' registrada en la base de datos nacional.")
        else:
            st.warning("⚠️ Todos los campos principales son obligatorios.")

with pestaña_buscar:
    st.subheader("Buscar Coincidencias Visuales")
    st.write("Subí la foto de un perro que hayas encontrado en la calle o que estés buscando para contrastarlo con la base de datos.")
    img_buscar_file = st.file_uploader("Subí la foto para buscar", type=["jpg", "jpeg", "png", "webp"], key="bus_img")
    
    if st.button("Buscar Coincidencias con IA"):
        if img_buscar_file:
            file_bytes = np.asarray(bytearray(img_buscar_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            huella_usuario = extraer_huella(img_bgr)
            
            if huella_usuario is None:
                st.error("❌ La IA no pudo detectar un perro en esta foto.")
            else:
                bd = cargar_base_datos()
                if not bd:
                    st.warning("📭 La base de datos nacional está vacía. Registrá un perro primero.")
                else:
                    mejor_coincidencia = None
                    mayor_porcentaje = 0.0
                    
                    for mascota in bd:
                        huella_db = np.array(mascota["huella"]).reshape(1, -1)
