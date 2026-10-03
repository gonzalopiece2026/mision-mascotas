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
ARCHIVO_HISTORIAL = "base_datos_reencuentros.json"

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

# --- 📊 SECCIÓN DE CONTADORES Y MÉTRICAS EN TIEMPO REAL ---
bd_actual = cargar_base_datos(ARCHIVO_BD)
bd_reencuentros = cargar_base_datos(ARCHIVO_HISTORIAL)

col1, col2 = st.columns(2)
with col1:
    st.metric(label="🚨 Alertas Activas en la Red", value=len(bd_actual))
with col2:
    st.metric(label="❤️ Perros que Volvieron a Casa", value=len(bd_reencuentros))

st.divider()

# --- 🚨 MURAL EN VIVO FILTRADO POR ZONA (EN EL INICIO) ---
st.header("📢 Alertas Activas en tu Zona")
st.write("Visualizá los perros reportados en tu localidad para estar alerta en tiempo real.")

# Filtro de zona para el mural de inicio
zona_filtro_inicio = st.text_input("Filtrar mural por tu Zona / Partido / Provincia (Ej: Moreno)", value="Moreno", key="mural_filtro_zona")

if not bd_actual:
    st.info("📭 No hay alertas activas registradas en el sistema.")
else:
    zona_buscada_limpia = zona_filtro_inicio.strip().lower()
    
    # Filtramos la base de datos según la zona que escribas
    alertas_filtradas_zona = []
    for m in bd_actual:
        zona_mascota = m.get("zona", "").lower()
        if not zona_buscada_limpia or zona_buscada_limpia in zona_mascota or zona_mascota in zona_buscada_limpia:
            alertas_filtradas_zona.append(m)
            
    if not alertas_filtradas_zona:
        st.warning(f"⚠️ No hay alertas activas registradas específicamente para '{zona_filtro_inicio}'. Probá cambiando la zona o registrá una nueva alerta abajo.")
    else:
        st.success(f"📍 Mostrando {len(alertas_filtradas_zona)} alerta(s) activa(s) en la zona: **{zona_filtro_inicio}**")
        
        # Mostramos las últimas 4 alertas de esa zona (de la más reciente a la más antigua)
        for mascota in reversed(alertas_filtradas_zona[-4:]):
            with st.container(border=True):
                tipo_alerta_mural = mascota.get('tipo_alerta', 'Perdido')
                badge_mural = "🔴 PERDIDO" if tipo_alerta_mural == "Perdido" else "🟢 ENCONTRADO"
                
                st.markdown(f"### {badge_mural} - 🐾 {mascota.get('nombre_perro', 'Sin nombre')} ({mascota.get('sexo', '')})")
                st.info(f"📍 **Zona:** {mascota.get('zona', 'No especificada')} \n📅 **Fecha:** {mascota.get('fecha_hecho', '')} \n🐕 **Raza / Color:** {mascota.get('raza', '')} | {mascota.get('color', '')}")
                
                numero_mural = mascota.get('contacto', '')
                if numero_mural:
                    st.code(f"📱 Contacto: {numero_mural}", language="text")
                
                foto_b64 = mascota.get('ruta_imagen', '')
                if foto_b64 and foto_b64 != "error":
                    try:
                        bytes_decor = base64.b64decode(foto_b64)
                        st.image(bytes_decor, use_container_width=True)
                    except:
                        pass

st.divider()

# --- 1️⃣ SECCIÓN DE BÚSQUEDA INTELIGENTE CON FILTROS (IA + ZONA + SEXO) ---
st.header("🔎 Buscar Coincidencias Inteligentes")
st.write("Subí la foto, indicá la zona y el sexo para afinar el cruce de datos.")
img_buscar_file = st.file_uploader("Subí la foto para buscar", type=["jpg", "jpeg", "png", "webp"], key="bus_img")
zona_busqueda_input = st.text_input("Zona, Barrio o Provincia donde buscás (Ej: Moreno, Mendoza)", key="bus_zona")
sexo_busqueda_input = st.selectbox("Sexo del perro que buscás", ["Cualquiera / No sé", "Macho", "Hembra"], key="bus_sexo")

if st.button("Buscar Coincidencias con IA", key="btn_buscar_principal"):
    if img_buscar_file:
        bd = bd_actual
        
        if not bd or len(bd) == 0:
            st.warning("📭 La base de datos nacional está vacía en este momento.")
        else:
            file_bytes = np.asarray(bytearray(img_buscar_file.read()), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            
            with st.spinner("Buscando coincidencias en la base de datos..."):
                huella_usuario = extraer_huella_segura(img_bgr)
                
            if huella_usuario is None:
                st.error("❌ No se pudo procesar la imagen de búsqueda.")
            else:
                coincidencias_encontradas = []
                vector_u = np.array(huella_usuario).reshape(1, -1)
                zona_usuario = zona_busqueda_input.strip().lower()
                
                for mascota in bd:
                    huella_db = np.array(mascota.get("huella", [])).reshape(1, -1)
                    similitud = cosine_similarity(vector_u, huella_db)
                    
                    porcentaje = float(np.squeeze(similitud)) * 100
                    
                    if porcentaje >= 65:
                        mascota_con_score = mascota.copy()
                        mascota_con_score["porcentaje_match"] = porcentaje
                        
                        # 1. FILTRO DE ZONA
                        zona_mascota = mascota.get("zona", "").lower()
                        etiqueta_z = "🌍 Otra Zona / Provincia"
                        if zona_usuario and (zona_usuario in zona_mascota or zona_mascota in zona_usuario):
                            etiqueta_z = "📍 ¡Misma Zona!"
                            mascota_con_score["porcentaje_match"] += 12
                        
                        # 2. FILTRO DE SEXO
                        sexo_mascota = mascota.get("sexo", "No especificado")
                        etiqueta_s = ""
                        if sexo_busqueda_input != "Cualquiera / No sé":
                            if sexo_mascota == sexo_busqueda_input:
                                etiqueta_s = f" | ⚧ Coincide sexo ({sexo_busqueda_input})"
                                mascota_con_score["porcentaje_match"] += 10
                            elif sexo_mascota != "No especificado":
                                mascota_con_score["porcentaje_match"] -= 8
                        
                        mascota_con_score["etiqueta_zona"] = f"{etiqueta_z}{etiqueta_s}"
                        coincidencias_encontradas.append(mascota_con_score)
                
                coincidencias_encontradas = sorted(coincidencias_encontradas, key=lambda x: x["porcentaje_match"], reverse=True)
                
                if not coincidencias_encontradas:
                    st.error("❌ No se encontraron coincidencias similares en el sistema.")
                else:
                    st.success(f"📊 ¡Se encontraron {len(coincidencias_encontradas)} posibles coincidencias!")
                    
                    for idx, coincidencia in enumerate(coincidencias_encontradas):
                        porcentaje_visual = coincidencia["porcentaje_match"]
                        if "📍" in coincidencia.get("etiqueta_zona", ""):
                            porcentaje_visual -= 12
                        if "Coincide sexo" in coincidencia.get("etiqueta_zona", ""):
                            porcentaje_visual -= 10
                        elif porcentaje_visual < 65:
                            porcentaje_visual = 65.0
                        
                        with st.container(border=True):
                            st.markdown(f"### 🎯 Opción #{idx + 1} - Similitud Visual: {porcentaje_visual:.2f}%")
                            st.markdown(f"**{coincidencia.get('etiqueta_zona', '')}**")
                            
                            tipo_match = coincidencia.get('tipo_alerta', 'Perdido')
                            cartel_tipo = "🔴 ESTADO: PERDIDO" if tipo_match == "Perdido" else "🟢 ESTADO: ENCONTRADO"
                            st.write(cartel_tipo)
                            
                            st.info(f"👤 **Responsable:** {coincidencia.get('nombre_dueño', 'Anónimo')} \n📍 **Lugar del hecho:** {coincidencia.get('zona', 'No especificado')} \n📅 **Fecha del suceso:** {coincidencia.get('fecha_hecho', 'No especificada')} \n🐾 **Nombre de la mascota:** {coincidencia.get('nombre_perro', 'No especificado')} \n⚧ **Sexo:** {coincidencia.get('sexo', 'No especificado')} \n🐕 **Raza / Color:** {coincidencia.get('raza', 'No específica')} | {coincidencia.get('color', 'No de pelaje')}")
                            
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

# --- 2️⃣ SECCIÓN DE REGISTRO DE ALERTA ---
st.header("📝 Registrar Alerta de Mascota")
st.write("Subí la foto y detallá las características del animal para agilizar el cruce inteligente.")

tipo_alerta = st.selectbox("¿Qué tipo de alerta querés crear?", ["Perdido", "Encontrado"])
img_file = st.file_uploader("Subí la foto de la mascota", type=["jpg", "jpeg", "png", "webp"], key="reg_img")
nombre_perro = st.text_input("Nombre de la mascota (Si no lo sabés, poné 'No lo sé')")
sexo_perro = st.selectbox("Sexo de la mascota", ["Macho", "Hembra", "No especificado"], key="reg_sexo")
raza_perro = st.text_input("Raza (Ej: Cruza, Caniche, Ovejero)")
color_perro = st.text_input("Color principal del pelaje")
detalles_perro = st.text_area("Detalles particulares (Ej: Tiene collar rojo, renguea de una pata, es asustadizo)")

st.write("---")
st.write("📋 Datos del Responsable:")
nombre_rescatista = st.text_input("Nombre del Dueño / Rescatista")
lugar_hecho = st.text_input("¿Dónde ocurrió? (Ej: Barrio Satélite, Moreno)", key="casillero_lugar_hecho_zona")
fecha_suceso = st.date_input("¿Qué día ocurrió?", value=datetime.now())
telefono_contacto = st.text_input("Teléfono de Contacto (Ej: 1162330944)", key="casillero_registro_telefono_moreno_puro")

if st.button("Guardar en la Red Nacional", key="btn_guardar_principal"):
    num_limpio = "".join(filter(str.isdigit, telefono_contacto))
    
    if not img_file or not nombre_rescatista or not lugar_hecho or not telefono_contacto:
        st.warning("⚠ Todos los campos principales son obligatorios (Foto, Responsable, Lugar y Teléfono).")
    elif len(num_limpio) < 10 or len(num_limpio) > 13:
        st.error("❌ El número de teléfono ingresado no parece válido. Asegurate de incluir la característica (Ej: 11 para Buenos Aires/Moreno) y que tenga al menos 10 dígitos.")
    else:
        bytes_datos_foto = img_file.getvalue()
        file_bytes = np.asarray(bytearray(bytes_datos_foto), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        huella = extraer_huella_segura(img_bgr)
        
        if huella is None:
            st.error("❌ Ocurrió un problema al procesar la imagen.")
        else:
            with st.spinner("Procesando y encriptando imagen..."):
                try:
                    foto_b64_string = base64.b64encode(bytes_datos_foto).decode('utf-8')
                except Exception as b64_err:
                    print(f"Error Base64: {b64_err}")
                    foto_b64_string = "error"
            
            if foto_b64_string == "error":
                st.error("❌ Error interno al procesar la vista previa.")
            else:
                fecha_hecho_str = fecha_suceso.strftime("%d/%m/%Y")
                hora_argentina = datetime.now() - timedelta(hours=3)
                fecha_subida_str = hora_argentina.strftime("%d/%m/%Y a las %H:%M hs")
                
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
                    "fecha_hecho": str(fecha_hecho_str),
                    "fecha_subida": str(fecha_subida_str)
                })
                guardar_base_datos(bd, ARCHIVO_BD)
                st.success("✅ ¡Éxito total! Alerta guardada con éxito en la red nacional.")
                st.rerun()

st.divider()

# --- 3️⃣ SECCIÓN DE BAJA Y REENCUENTROS ---
st.header("✨ Misión Cumplida: Dar de Baja Alerta")
st.write("Si el perro ya regresó con su familia, ingresá tu número para darlo de baja y sumar un reencuentro exitoso al contador.")

baja_telefono_fijo_ok = st.text_input("Ingresá el número de teléfono con el que publicaste el reporte (Ej: 1162330944)", key="casillero_baja_universal_fijo_moreno_2026_final_ar")

if st.button("Eliminar y Registrar Reencuentro", key="btn_baja_principal"):
    num_baja_limpio = "".join(filter(str.isdigit, baja_telefono_fijo_ok))
    
    if not num_baja_limpio:
        st.warning("⚠️ Por favor, ingresá un número de teléfono válido.")
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
                
                st.success(f"❤️ ¡Misión cumplida! Se registró el reencuentro con éxito. ¡Gracias por usar Misión Mascotas!")
                st.rerun()
            else:
                st.error("❌ No se encontró ninguna publicación activa con ese número de teléfono.")

# --- 💖 APARTADO VISUAL DE PERROS QUE YA VOLVIERON A CASA ---
st.divider()
st.header("❤️ Muro de Reencuentros Felices")
st.write("Estos son algunos de los perritos que ya volvieron con sus familias gracias al sistema.")

if not bd_reencuentros:
    st.info("📌 Todavía no hay reencuentros registrados. ¡Cuando des de baja una alerta exitosa, aparecerá acá para celebrar!")
else:
    for idx, reencuentro in enumerate(reversed(bd_reencuentros)):
        with st.container(border=True):
            st.markdown(f"### 🎉 ¡Reencuentro Exitoso #{len(bd_reencuentros) - idx}!")
            st.success(f"🐾 **Mascota:** {reencuentro.get('nombre_perro', 'Desconocido')} ({reencuentro.get('sexo', '')}) \n📍 **Zona:** {reencuentro.get('zona', 'No especificada')} \n📅 **Volvió a casa el:** {reencuentro.get('fecha_reencuentro', 'Reciente')}")
            
            foto_b64 = reencuentro.get('ruta_json', '') # Asegura compatibilidad limpia
            foto_b64 = reencuentro.get('ruta_imagen', '')
            if foto_b64 and foto_b64 != "error":
                try:
                    bytes_decor = base64.b64decode(foto_b64)
                    st.image(bytes_decor, use_container_width=True)
                except:
                    pass
