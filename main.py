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

# --- INTERFAZ GRÁFICA NATIVA ---
pestaña_buscar, pestaña_registrar, pestaña_robot, pestaña_donar = st.tabs([
    "🔎 BUSCAR", 
    "📝 ALERTA", 
    "🤖 ROBOT",
    "💝 DONAR"
])

with pestaña_donar:
    st.subheader("💝 Apoyá a Misión Mascotas")
    st.write("Esta plataforma es 100% gratuita y libre de publicidad para ayudar a que más familias vuelvan a encontrarse.")
    st.write("Tu donación nos ayuda directamente a mantener los servidores online las 24 horas y seguir mejorando la Inteligencia Artificial.")
    
    st.divider()
    
    st.markdown("### 💳 Transferencia Directa (Monto Libre)")
    # Cuando quieras podés cambiar estos dos renglones por tu Alias y CBU reales del banco
    st.text("Alias: TU.ALIAS.REAL.AQUÍ")
    st.text("CBU / CVU: 0000000000000000000000")
    st.text("Titular: Gonzalo")
    
    st.divider()
    
    st.markdown("### 🚀 Links de Mercado Pago rápidos")
    st.write("Elegí una opción rápida o usá el botón de colaboración libre:");
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button("☕ Cafecito ($1.000)", "https://link-de-mercado-pago-de-1000")
    with col2:
        st.link_button("🍔 Combo ($3.000)", "https://link-de-mercado-pago-de-3000")
    with col3:
        st.link_button("💎 Súper ($5.000)", "https://link-de-mercado-pago-de-5000")
        
    st.write("") 
    # ¡BOTÓN INTEGRADO CON TU LINK OFICIAL DE MONTO LIBRE!
    st.link_button("✨ COLABORACIÓN CON MONTO LIBRE", "https://link.mercadopago.com.ar/misionmascotas")

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
    st.write("Subí la foto de un perro para contrastarlo con la base de datos.")
    img_buscar_file = st.file_uploader("Subí la foto para buscar", type=["jpg", "jpeg", "png", "webp"], key="bus_img")
    if st.button("Buscar Coincidencias con IA"):
        if img_buscar_file:
            file_bytes = np.asarray(bytearray(img_buscar_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            
            with st.spinner("Buscando en la base de datos..."):
                huella_usuario = extraer_huella(img_bgr)
                
            if huella_usuario is None:
                st.error("❌ La IA no pudo detectar un perro en esta foto. Asegurate de que el perrito esté bien visible.")
            else:
                bd = cargar_base_datos()
                if not bd:
                    st.warning("📭 La base de datos nacional está vacía. Registrá un perro primero.")
                else:
                    mejor_coincidencia = None
                    mayor_porcentaje = 0.0
                    for mascota in bd:
                        huella_db = np.array(mascota["huella"]).reshape(1, -1)
                        vector_u = np.array(huella_usuario).reshape(1, -1)
                        similitud = cosine_similarity(vector_u, huella_db)
                        
                        porcentaje = float(similitud) * 100
                        if porcentaje > mayor_porcentaje:
                            mayor_porcentaje = porcentaje
                            mejor_coincidencia = mascota
                            
                    if mejor_coincidencia and mayor_porcentaje > 65:
                        st.success(f"📊 ¡COINCIDENCIA ENCONTRADA CON ÉXITO! ({mayor_porcentaje:.2f}% de parecido)")
                        
