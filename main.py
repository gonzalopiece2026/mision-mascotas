import os
import cv2
import json
import numpy as np
import streamlit as st
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURACIÓN DE LA PÁGINA WEB ---
st.set_page_config(page_title="Misión Mascotas Pro", page_icon="🐶", layout="centered")

st.markdown("# 🐶 Misión Mascotas - IA Buscador")
st.markdown("Registrá alertas de redes sociales y buscá coincidencias al instante.")

# Cargamos la IA y la base de datos
@st.cache_resource
def cargar_modelo():
    return YOLO("yolov8n.pt")

modelo = cargar_modelo()
ARCHIVO_BD = "base_datos_mascotas.json"

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
                coordenadas = box.xyxy.tolist()[0]
                x1, y1, x2, y2 = map(int, coordenadas)
                perro_recortado = img_bgr[y1:y2, x1:x2]
                img_redim = cv2.resize(perro_recortado, (64, 64))
                gris = cv2.cvtColor(img_redim, cv2.COLOR_BGR2GRAY)
                return (gris.flatten() / 255.0).tolist()
    return None

# --- CREACIÓN DE LAS PESTAÑAS EN LA WEB ---
pestaña_buscar, pestaña_registrar = st.tabs(["🔎 BUSCADOR INTELIGENTE", "📝 REGISTRAR NUEVA ALERTA"])

with pestaña_registrar:
    st.subheader("Registrar Perro Perdido/Encontrado de Redes")
    img_file = st.file_uploader("Subí la foto del perro", type=["jpg", "jpeg", "png", "webp"], key="reg_img")
    nombre = st.text_input("Nombre del Dueño / Persona que publica")
    zona = st.text_input("Zona / Localidad (Ej: Ituzaingó)")
    contacto = st.text_input("Teléfono / Celular de Contacto")
    link = st.text_input("Link de Facebook o Instagram (Opcional)")
    
    if st.button("Guardar en Base de Datos"):
        if img_file and nombre and zona and contacto:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            huella = extraer_huella(img_bgr)
            
            if huella is None:
                st.error("❌ La IA no detectó ningún perro en la foto.")
            else:
                os.makedirs("fotos_registradas", exist_ok=True)
                ruta_foto = f"fotos_registradas/perro_{nombre}_{contacto}.jpg"
                cv2.imwrite(ruta_foto, img_bgr)
                
                bd = cargar_base_datos()
                bd.append({
                    "nombre_dueño": nombre, "zona": zona, "contacto": contacto,
                    "link_redes": link if link else "No especificado",
                    "ruta_imagen": ruta_foto, "huella": huella
                })
                guardar_base_datos(bd)
                st.success(f"✅ ¡Éxito! Mascota de '{nombre}' registrada correctamente.")
        else:
            st.warning("⚠️ Todos los campos son obligatorios.")

with pestaña_buscar:
    st.subheader("Buscar Coincidencias con IA")
    img_buscar_file = st.file_uploader("Subí la foto para buscar", type=["jpg", "jpeg", "png", "webp"], key="bus_img")
    
    if st.button("Buscar Coincidencias"):
        if img_buscar_file:
            file_bytes = np.asarray(bytearray(img_buscar_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            huella_usuario = extraer_huella(img_bgr)
            
            if huella_usuario is None:
                st.error("❌ La IA no pudo detectar un perro en esta foto.")
            else:
                bd = cargar_base_datos()
                if not bd:
                    st.warning("📭 La base de datos está vacía. Registrá un perro primero.")
                else:
                    mejor_coincidencia = None
                    mayor_porcentaje = 0.0
                    
                    for mascota in bd:
                        huella_db = np.array(mascota["huella"]).reshape(1, -1)
                        vector_u = np.array(huella_usuario).reshape(1, -1)
                        similitud = cosine_similarity(vector_u, huella_db)
                        porcentaje = float(similitud[0][0]) * 100
                        
                        if porcentaje > mayor_porcentaje:
                            mayor_porcentaje = porcentaje
                            mejor_coincidencia = mascota
                            
                    if mejor_coincidencia and mayor_porcentaje > 65:
                        st.info(f"📊 ¡COINCIDENCIA ENCONTRADA! ({mayor_porcentaje:.2f}% de parecido)")
                        st.write(f"👤 **Dueño/Publicado por:** {mejor_coincidencia['nombre_dueño']}")
                        st.write(f"📍 **Zona:** {mejor_coincidencia['zona']}")
                        st.write(f"📞 **Contacto:** {mejor_coincidencia['contacto']}")
                        st.write(f"🔗 **Link:** {mejor_coincidencia['link_redes']}")
                        
                        img_res = cv2.imread(mejor_coincidencia["ruta_imagen"])
                        st.image(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB), caption="Foto en el sistema")
                    else:
                        st.error("❌ No se encontraron perros similares con alto porcentaje.")
