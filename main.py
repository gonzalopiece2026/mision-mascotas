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
            if not bd or len(bd) == 0:
                st.warning("📭 La base de datos nacional está vacía en este momento. Registrá una mascota abajo primero para poder realizar búsquedas.")
            else:
                coincidencias_encontradas = []
                vector_u = np.array(huella_usuario).reshape(1, -1)
                
                for mascota in bd:
                    huella_db = np.array(mascota["huella"]).reshape(1, -1)
                    similitud = cosine_similarity(vector_u, huella_db)
                    porcentaje = float(np.ravel(similitud)) * 100
                    
                    if porcentaje >= 65:
                        mascota_con_score = mascota.copy()
                        mascota_con_score["porcentaje_match"] = porcentaje
                        coincidencias_encontradas.append(mascota_con_score)
                
                coincidencias_encontradas = sorted(coincidencias_encontradas, key=lambda x: x["porcentaje_match"], reverse=True)
                
                if not coincidencias_encontradas:
                    st.error("❌ No se encontraron coincidencias similares en el sistema.")
                else:
                    st.success(f"📊 ¡Se encontraron {len(coincidencias_encontradas)} posibles coincidencias en la red nacional!")
                    
                    for idx, coincidencia in enumerate(coincidencias_encontradas):
                        porcentaje_actual = coincidencia["porcentaje_match"]
                        
                        with st.container(border=True):
                            st.markdown(f"### 🎯 Opción #{idx + 1} - Coincidencia del {porcentaje_actual:.2f}%")
                            
                            tipo_match = coincidencia.get('tipo_alerta', 'Perdido')
                            cartel_tipo = "🔴 ESTADO: PERDIDO" if tipo_match == "Perdido" else "🟢 ESTADO: ENCONTRADO"
                            st.write(cartel_tipo)
                            
                            st.info(f"👤 **Responsable:** {coincidencia.get('nombre_dueño', 'Anónimo')} \n📍 **Lugar del hecho:** {coincidencia.get('zona', 'No especificado')} \n📅 **Fecha del suceso:** {coincidencia.get('fecha_hecho', 'No especificada')} \n🐾 **Nombre de la mascota:** {coincidencia.get('nombre_perro', 'No especificado')} \n🐕 **Raza / Color:** {coincidencia.get('raza', 'No específica')} | {coincidencia.get('color', 'No especificado')} \n📝 **Detalles particulares:** {coincidencia.get('detalles', 'Sin detalles adicionales')}")
                            
                            numero_match = coincidencia.get('contacto', '')
                            st.markdown("**📱 Teléfono de Contacto:**")
                            st.code(f"{numero_match}", language="text")
                            
                            foto_b64 = coincidencia.get('ruta_imagen', '')
                            if foto_b64 and foto_b64 != "error":
                                try:
                                    bytes_decor = base64.b64decode(foto_b64)
                                    st.image(bytes_decor, use_container_width=True)
                                except:
                                    st.text("📷 Foto no compatible")
                            st.write("")
    else:
        st.warning("⚠️ Primero tenés que subir una foto en el recuadro de arriba para poder buscar.")

st.divider()

# --- 2️⃣ SECCIÓN DE BAJA INDESTRUCTIBLE UBICADA ESTRATÉGICAMENTE EN EL CENTRO ---
st.header("✨ Misión Cumplida: Dar de Baja Alerta")
st.write("Si el perro ya regresó con su familia o apareció el dueño, ingresá tu número de teléfono celular para remover las publicaciones de la nube.")

baja_telefono_fijo_ok = st.text_input("Ingresá el número de teléfono con el que publicaste el reporte (Ej: 1162330944)", key="casillero_baja_universal_fijo_moreno_2026_exclusivo")

if st.button("Desactivar Mis Alertas Permanentemente", key="btn_baja_sistema_universal_moreno_ok_final"):
    if baja_telefono_fijo_ok:
        tel_baja_limpio = "".join(filter(str.isdigit, baja_telefono_fijo_ok))
        bd_actual = cargar_base_datos()
        
        cantidad_removidos = 0
        nueva_bd_filtrada = []
        
        for m in bd_actual:
            if m.get("contacto", "") == tel_baja_limpio:
                cantidad_removidos += 1
            else:
                nueva_bd_filtrada.append(m)
                
        if cantidad_removidos > 0:
            guardar_base_datos(nueva_bd_filtrada)
            st.success(f"🎉 ¡Felicidades! Se removieron con éxito {cantidad_removidos} publicación(es) de la red nacional.")
            st.balloons()
            st.rerun()
        else:
            st.error("❌ No se encontró ninguna alerta activa registrada con ese número de teléfono celular.")
    else:
        st.warning("⚠️ Debes ingresar el número de teléfono con el que realizaste la publicación para validar la baja.")

st.divider()

# --- 3️⃣ SECCIÓN DE REGISTRO DE ALERTA ---
st.header("📝 Registrar Alerta de Mascota")
st.write("Subí la foto y detallá las características del animal para agilizar el cruce inteligente.")

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
telefono_contacto = st.text_input("Teléfono de Contacto (Ej: 1162330944)", key="casillero_registro_telefono_moreno_puro")

if st.button("Guardar en la Red Nacional", key="btn_guardar_principal"):
    if img_file and nombre_rescatista and lugar_hecho and telefono_contacto:
        num_limpio = "".join(filter(str.isdigit, telefono_contacto))
                
        # Leemos los bytes del cargador de archivos de internet
        bytes_datos_foto = img_file.getvalue()
        
        file_bytes = np.asarray(bytearray(bytes_datos_foto), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        
        huella = extraer_huella_segura(img_bgr)
        
        if huella is None:
            st.error("❌ Ocurrió un problema al procesar la imagen.")
        else:
            with st.spinner("Procesando y encriptando imagen de forma nativa viva..."):
                try:
                    # CORRECCIÓN MAESTRA: Codificamos de forma directa los bytes puros para evitar fallas de RAM en la nube
                    foto_b64_string = base64.b64encode(bytes_datos_foto).decode('utf-8')
                except Exception as b64_err:
                    print(f"Error Base64: {b64_err}")
                    foto_b64_string = "error"
            
            if foto_b64_string == "error":
                st.error("❌ Error interno al procesar la vista previa. Intenta con otra imagen.")
            else:
                fecha_hecho_str = fecha_suceso.strftime("%d/%m/%Y")
                hora_argentina = datetime.now() - timedelta(hours=3)
                fecha_subida_str = hora_argentina.strftime("%d/%m/%Y a las %H:%M hs")
                
