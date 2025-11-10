# main.py
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple
import bcrypt
import httpx

# Fastapi
from fastapi import FastAPI, File, Form, Path, Query, UploadFile, HTTPException, Depends, status 
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# BD
from database import create_db_and_tables, get_session # Importar funciones de database.py
from models import Trabajador, LecturaEstres, Sesion, Cuestionario, TrabajadorJefe #Importa models y sus entidades
from sqlalchemy.ext.asyncio import AsyncSession # Importar AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select
from pydantic import BaseModel, Field
import logging

# Modelo DL y preprocesamiento de imagenes
import torch
from torchvision import transforms 
from PIL import Image
from io import BytesIO
import os
import sys
from sqlalchemy import delete, text, func




# MODEL_SERVER_URL = "http://127.0.0.1:8001/predict/"
MODEL_SERVER_URL = os.getenv(
    "MODEL_SERVER_URL",
    "http://127.0.0.1:8001/predict/"  # opcional: valor por defecto
)


# --- CONFIGURACIÓN PARA DETECTAR EL MODELO ---
#MODEL_DIR = os.path.dirname(__file__)
#MODEL_PATH = os.path.join(MODEL_DIR, "stress.pth")
#print(f"Tamaño del archivo: {os.path.getsize(MODEL_PATH)} bytes")
# CLASSES = ["No Estrés", "Estrés"]

# Transformaciones con torchvision 
#transform = transforms.Compose([
#    transforms.Resize((224, 224)), # Cambiar resolucion
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 
#])

class UserCreate(BaseModel):
    nombre: str
    username: str
    correo: str
    password: str
    fecha_de_nacimiento: Optional[date] = None
    genero: Optional[str] = None
    estado_civil: Optional[str] = None
    uso_de_anteojos: Optional[bool] = False
    estudio_y_trabajo: Optional[str] = None
    horas_trabajo_semanal: Optional[int] = None
    horas_descanso_dia: Optional[int] = None
    
    
class AdminRoleUpdate(BaseModel):
    role_id: int  # 1=Usuario, 2=Admin, 3=Jefe

class AsignacionCreate(BaseModel):
    jefe_id: int
    subordinado_ids: List[int]

class AsignacionDelete(BaseModel):
    jefe_id: int
    subordinado_id: int


class TrabajadorBasic(BaseModel):
    trabajador_id: int
    nombre: str
    username: str
    role_id: Optional[int] = None  # ✅ acepta None

class TrabajadorPublic(BaseModel):
    trabajador_id: int
    nombre: str
    username: str
    fecha_de_nacimiento: Optional[date] = None
    genero: Optional[str] = None
    estado_civil: Optional[str] = None
    uso_de_anteojos: Optional[bool] = None
    estudio_y_trabajo: Optional[str] = None
    horas_trabajo_semanal: Optional[int] = None
    horas_descanso_dia: Optional[int] = None
    created_at: datetime
    updated_at: datetime


# main.py

class TrabajadorPublic(BaseModel):
    trabajador_id: int
    nombre: str
    username: str
    fecha_de_nacimiento: Optional[date] = None
    genero: Optional[str] = None
    estado_civil: Optional[str] = None
    uso_de_anteojos: Optional[bool] = None
    estudio_y_trabajo: Optional[str] = None
    horas_trabajo_semanal: Optional[int] = None
    horas_descanso_dia: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    
class SesionSummaryResponse(BaseModel): # O puedes usar Pydantic BaseModel si prefieres
    sesion_id: int
    trabajador_id: int
    fecha_sesion: date
    hora_inicio: datetime
    hora_fin: Optional[datetime] = None
    estado_grabacion: Optional[str] = None
    duracion_calculada_segundos: Optional[float] = None # Usamos float por si hay fracciones de segundo
    porcentaje_estres: Optional[float] = None
    total_lecturas: int
    cuestionario: Optional[Cuestionario] = None 


class UserLogin(BaseModel):
    username: str
    password: str
    
class SessionStartRequest(BaseModel):
    trabajador_id: int
    
class SessionEndRequest(BaseModel):
    hora_fin: Optional[datetime] = Field(default=None)
    estado_grabacion: Optional[str] = Field(default="Finalizada")
    duracion_grabacion_segundos: Optional[int] = Field(default=None)
    nivel_estres_modelo_avg: Optional[float] = Field(default=None)
    distancia_rostro_pantalla_avg: Optional[float] = Field(default=None)
    brillo_pantalla_avg: Optional[float] = Field(default=None)

class DailyStressSummary(BaseModel):
    fecha: date
    porcentaje_estres_promedio: Optional[float] = None
    duracion_total_grabacion_segundos: Optional[float] = None
    numero_sesiones: int
    
class TrabajadorUpdate(BaseModel):
    nombre: Optional[str] = None
    fecha_de_nacimiento: Optional[date] = None
    genero: Optional[str] = None
    estado_civil: Optional[str] = None
    uso_de_anteojos: Optional[bool] = None
    estudio_y_trabajo: Optional[str] = None
    horas_trabajo_semanal: Optional[int] = None
    horas_descanso_dia: Optional[int] = None

class MonthlyOverallSummary(BaseModel):
    trabajador_id: int
    month: int
    year: int
    tiempo_total_grabacion_mes_segundos: float
    tiempo_promedio_grabacion_por_dia_activo_segundos: Optional[float] = None # Días con al menos una sesión
    nivel_estres_promedio_mensual: Optional[float] = None # Promedio de los porcentajes de estrés de las sesiones
    numero_total_sesiones_mes: int
    promedio_sesiones_por_dia_activo: Optional[float] = None # Días con al menos una sesión
    dias_con_actividad: int

class CuestionarioCreate(BaseModel):
    sesion_id: int
    descripcion_trabajo: Optional[str] = None
    nivel_de_sensacion_estres: Optional[int] = None
    molestias_fisicas_visual: Optional[int] = None
    molestias_fisicas_otros: Optional[int] = None
    dificultad_concentracion: Optional[int] = None
    
'''
# --- CARGAR EL MODELO ---
model = None # modelo personalizado que hereda nn.module
try:
    print(f"Intentando cargar el modelo PyTorch (nn.Module) desde: {MODEL_PATH}")
    # map_location para asegurar que se carga en CPU.
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)

    # Verificar si el objeto cargado es un modelo
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"El archivo cargado no es un nn.Module. Tipo encontrado: {type(model)}")

    model.eval() # Poner modo evaluación
    print(f"Modelo PyTorch (nn.Module) cargado exitosamente desde {MODEL_PATH}")

except FileNotFoundError:
     print(f"ERROR CRÍTICO - MODELO NO ENCONTRADO: {MODEL_PATH}")
     model = None
except ModuleNotFoundError as e_module:
    print(f"ERROR CRÍTICO - MODULO FALTANTE AL CARGAR MODELO: {e_module}")
    print("Esto significa que la definición de alguna capa o clase en tu modelo guardado no se encuentra.")
    print("Asegúrate de tener todas las bibliotecas necesarias (incluida fastai si alguna capa es de fastai).")
    model = None  
except RuntimeError as e_runtime:
    print(f"ERROR CRÍTICO - RUNTIME ERROR AL CARGAR MODELO: {e_runtime}")
    import traceback
    traceback.print_exc()
    model = None
except Exception as e_general:
    print(f"ERROR CRÍTICO - ERROR GENERAL AL CARGAR MODELO: {e_general}")
    import traceback
    traceback.print_exc()
    model = None
'''

# --- CONFIGURAR FastAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el backend
@app.on_event("startup")
async def on_startup():
    print("Iniciando aplicación FastAPI...")
    await create_db_and_tables()
    print("Tablas de la base de datos verificadas/creadas.")
    
    
    

@app.post("/register/", response_model=TrabajadorPublic, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate, # Usar el modelo Pydantic para la validación de entrada
    session: AsyncSession = Depends(get_session)
):
    logging.info(f"Recibida petición de registro para el usuario: {user_data.username}")

    # Hashear la contraseña antes de guardarla en la base de datos
    # bcrypt.gensalt() genera un salt aleatorio
    # .decode('utf-8') es necesario porque hashpw devuelve bytes
    hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    # Crear una nueva instancia del modelo Trabajador con la contraseña hasheada
    new_trabajador = Trabajador(
        nombre=user_data.nombre,
        username=user_data.username,
        correo=user_data.correo.strip().lower(),  # ✅ normalizado
        password=hashed_password, # Almacenar la contraseña hasheada
        fecha_de_nacimiento=user_data.fecha_de_nacimiento,
        genero=user_data.genero,
        estado_civil=user_data.estado_civil,
        uso_de_anteojos=user_data.uso_de_anteojos,
        estudio_y_trabajo=user_data.estudio_y_trabajo,
        horas_trabajo_semanal=user_data.horas_trabajo_semanal,
        horas_descanso_dia=user_data.horas_descanso_dia,
        created_at=datetime.now(timezone.utc), # Establecer created_at manualmente
        updated_at=datetime.now(timezone.utc)  # Establecer updated_at manualmente
    )

    try:
        session.add(new_trabajador)
        await session.commit()
        await session.refresh(new_trabajador)
        return new_trabajador
    except IntegrityError as e:
        await session.rollback()
        if "username" in str(e.orig):
            detail = "El nombre de usuario o correo ya existe."
        elif "correo" in str(e.orig):
            detail = "El nombre de usuario o correo ya existe."
        else:
            detail = e
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail
        )
    except Exception as e:
        # Captura cualquier otro error inesperado durante la creación.
        await session.rollback()
        print(f"Error inesperado al crear usuario: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al crear la cuenta de usuario."
        )

# --- NUEVO ENDPOINT: Iniciar sesión ---
@app.post("/login/")
async def login_user(
    user_data: UserLogin, # Usar el modelo Pydantic para la validación de entrada
    session: AsyncSession = Depends(get_session)
):
    # Buscar al trabajador por nombre de usuario
    statement = select(Trabajador).where(Trabajador.username == user_data.username)
    result = await session.execute(statement)
    trabajador = result.scalars().first() # Obtener el primer resultado (o None si no se encuentra)

    if not trabajador:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Credenciales inválidas." # Mensaje genérico por seguridad
        )

    # Verificar la contraseña hasheada
    # bcrypt.checkpw compara la contraseña en texto plano con el hash almacenado
    if not bcrypt.checkpw(user_data.password.encode('utf-8'), trabajador.password.encode('utf-8')):
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Credenciales inválidas." # Mensaje genérico por seguridad
        )

    # Si la autenticación es exitosa, puedes devolver un mensaje de éxito
    # En una aplicación real, aquí generarías y devolverías un Token de Autenticación (JWT)
    return {"message": "Inicio de sesión exitoso", "trabajador_id": trabajador.trabajador_id}


# --- ENDPOINT DE PREDICCIÓN ---
@app.post("/predict/")
async def predict_stress(
    file: UploadFile = File(...),
    sesion_id: int = Form(...),
    session: AsyncSession = Depends(get_session)
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo subido debe ser una imagen.")

    contents = await file.read()

    # >>> AQUI: incluir sesion_id en los datos del form <<<
    data = {'sesion_id': str(sesion_id)}
    files = {'file': (file.filename, contents, file.content_type)}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(MODEL_SERVER_URL, data=data, files=files, timeout=10.0)
            response.raise_for_status()
            prediction_data = response.json()
            predicted_class_name = prediction_data.get("prediction")
            confidence = prediction_data.get("confidence")
            # (Opcional) Si quieres, extrae storage.gs_uri para guardarlo en tu BD
            storage_info = prediction_data.get("storage", {})
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="No se pudo conectar con el servicio de predicción.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error en el servicio de predicción: {e.response.text}")
    except Exception:
        raise HTTPException(status_code=500, detail="Error interno al procesar la predicción.")

    # Guardar lectura en la BD (como ya lo tienes)
    try:
        current_utc_time = datetime.now(timezone.utc)
        lectura_individual = LecturaEstres(
            sesion_id=sesion_id,
            prediccion=predicted_class_name,
            confianza=round(confidence, 4),
            timestamp_lectura=current_utc_time,
            created_at=current_utc_time,
            updated_at=current_utc_time
        )
        session.add(lectura_individual)
        await session.commit()
        await session.refresh(lectura_individual)
    except Exception:
        raise HTTPException(status_code=500, detail="La predicción se realizó pero no se pudo guardar en la base de datos.")

    return JSONResponse(content={
        "prediction": predicted_class_name,
        "confidence": round(confidence, 4),
        "lectura_estres_id": lectura_individual.lectura_estres_id,
        # (Opcional) expón info de Storage para trazabilidad/dataset
        "storage": storage_info
    })



@app.post("/sessions/start/", response_model=Sesion)
async def start_session(
    request_data: SessionStartRequest,
    session: AsyncSession = Depends(get_session)
):
    current_utc_time = datetime.now(timezone.utc)
    new_session = Sesion(
        trabajador_id=request_data.trabajador_id,
        fecha_sesion=current_utc_time.date(), # Solo la fecha
        hora_inicio=current_utc_time,
        hora_fin=None,
        estado_grabacion="Iniciada",
        created_at=current_utc_time,
        updated_at=current_utc_time
    )
    try:
        session.add(new_session)
        await session.commit()
        await session.refresh(new_session)
        return new_session
    except Exception as e:
        print(f"Error al iniciar sesión: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error al iniciar la sesión de grabación.")


# --- NUEVO ENDPOINT: Finalizar una sesión de grabación ---
@app.put("/sessions/{sesion_id}/end/", response_model=Sesion)
async def end_session(
    request_data: SessionEndRequest, # Movido antes de los parámetros con default
    sesion_id: int = Path(..., description="ID de la sesión a finalizar"),
    session: AsyncSession = Depends(get_session)
):
    current_utc_time = datetime.now(timezone.utc)
    statement = select(Sesion).where(Sesion.sesion_id == sesion_id)
    result = await session.execute(statement)
    db_session = result.scalars().first()

    if not db_session:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Sesión no encontrada.")

    # Calcular duración si hora_fin no se proporciona o si hora_inicio existe
    final_hora_fin = request_data.hora_fin if request_data.hora_fin else current_utc_time
    duracion = None
    if db_session.hora_inicio and final_hora_fin:
        # Asegurarse de que ambos datetimes sean timezone-aware para la resta
        if db_session.hora_inicio.tzinfo is None:
            db_session.hora_inicio = db_session.hora_inicio.replace(tzinfo=timezone.utc)
        if final_hora_fin.tzinfo is None:
            final_hora_fin = final_hora_fin.replace(tzinfo=timezone.utc)
        
        duration_delta = final_hora_fin - db_session.hora_inicio
        duracion = int(duration_delta.total_seconds())

    db_session.hora_fin = final_hora_fin
    db_session.estado_grabacion = request_data.estado_grabacion if request_data.estado_grabacion else "Finalizada"
    db_session.duracion_grabacion_segundos = duracion
    db_session.nivel_estres_modelo_avg = request_data.nivel_estres_modelo_avg
    db_session.distancia_rostro_pantalla_avg = request_data.distancia_rostro_pantalla_avg
    db_session.brillo_pantalla_avg = request_data.brillo_pantalla_avg
    db_session.updated_at = current_utc_time # Actualizar timestamp de actualización

    try:
        session.add(db_session) # Add para que SQLAlchemy detecte los cambios
        await session.commit()
        await session.refresh(db_session)
        return db_session
    except Exception as e:
        print(f"Error al finalizar sesión {sesion_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error al finalizar la sesión de grabación.")
# --- NUEVO ENDPOINT: Eliminar una sesión (delete lógico) ---
@app.delete("/sessions/{sesion_id}/delete", response_model=Sesion)
async def delete_session(
    sesion_id: int = Path(..., description="ID de la sesión a eliminar"),
    session: AsyncSession = Depends(get_session)
):
    current_utc_time = datetime.now(timezone.utc)
    statement = select(Sesion).where(Sesion.sesion_id == sesion_id)
    result = await session.execute(statement)
    db_session = result.scalars().first()

    if not db_session:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Sesión no encontrada."
        )

    # Marcamos solo como Eliminado
    db_session.estado_grabacion = "Eliminado"
    db_session.updated_at = current_utc_time

    try:
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        return db_session
    except Exception as e:
        print(f"Error al eliminar sesión {sesion_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Error al eliminar la sesión."
        )



@app.get("/trabajadores/")
async def get_all_trabajadores(session: AsyncSession = Depends(get_session)): # Usar AsyncSession
    """
    Obtiene y devuelve una lista de todos los trabajadores de la base de datos.
    """
    try:
        # Crea una sentencia para seleccionar todos los registros de la tabla trabajadores
        statement = select(Trabajador)
        # Ejecuta la sentencia con session.execute() para AsyncSession
        results = await session.execute(statement)
        # Obtiene los resultados usando .scalars().all() para AsyncSession
        trabajadores = results.scalars().all()
        return trabajadores
    except Exception as e:
        print(f"Error al obtener trabajadores: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error al obtener la lista de trabajadores.")

@app.get("/statistics/", response_model=List[LecturaEstres]) # Renombrado para claridad
async def get_lecturas_estres(
    sesion_id: Optional[int] = None, # Parámetro de consulta opcional
    db_session: AsyncSession = Depends(get_session) # Renombrado el parámetro de la función
):
    statement = select(LecturaEstres)
    if sesion_id is not None:
        statement = statement.where(LecturaEstres.sesion_id == sesion_id)
    
    results = await db_session.execute(statement)
    lecturas = results.scalars().all()
    return lecturas



@app.get("/sesiones/")
async def get_statistics(session: AsyncSession = Depends(get_session)): # Usar AsyncSession
    statement = select(Sesion)
    
    results = await session.execute(statement)

    sesiones = results.scalars().all()

    return sesiones
   
@app.get("/sesiones/{sesion_id}/summary/", response_model=SesionSummaryResponse)
async def get_session_summary(
    sesion_id: int, # Parámetro de ruta
    db_session: AsyncSession = Depends(get_session)
):
    # 1. Obtener la información de la sesión
    sesion_statement = select(Sesion).where(Sesion.sesion_id == sesion_id)
    sesion_result = await db_session.execute(sesion_statement)
    sesion_obj = sesion_result.scalar_one_or_none()

    if not sesion_obj:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    # 2. Obtener todas las lecturas de estrés para esta sesión
    lecturas_statement = select(LecturaEstres).where(LecturaEstres.sesion_id == sesion_id).order_by(LecturaEstres.timestamp_lectura)
    lecturas_result = await db_session.execute(lecturas_statement)
    lecturas = lecturas_result.scalars().all()
    
    total_lecturas = len(lecturas)
    porcentaje_estres = None
    duracion_calculada_segundos = None

    # 3. Calcular porcentaje de estrés
    if total_lecturas > 0:
        conteo_estres = sum(1 for lec in lecturas if lec.prediccion == "Estrés")
        porcentaje_estres = (conteo_estres / total_lecturas) * 100
    else:
        porcentaje_estres = 0.0 # O None si prefieres no mostrar 0% para sesiones sin lecturas

    # 4. Calcular duración
    # Prioridad 1: Usar hora_fin y hora_inicio de la sesión si está finalizada
    if sesion_obj.estado_grabacion == "Finalizada" and sesion_obj.hora_fin and sesion_obj.hora_inicio:
        # Asegurarse de que ambos son conscientes de la zona horaria o ambos son naive para la resta
        # Si vienen de la DB con timezone=True, deberían estar bien.
        inicio_ts = sesion_obj.hora_inicio
        fin_ts = sesion_obj.hora_fin
        
        duracion_delta = fin_ts - inicio_ts
        duracion_calculada_segundos = duracion_delta.total_seconds()

    # Prioridad 2: Calcular duración a partir de los timestamps de las lecturas si la sesión no está finalizada o falta hora_fin
    elif total_lecturas > 0:
        # Asumimos que las lecturas están ordenadas por timestamp_lectura
        primera_lectura_ts = lecturas[0].timestamp_lectura
        ultima_lectura_ts = lecturas[-1].timestamp_lectura
        
        # Mismo cuidado con timezones si es necesario
        duracion_delta = ultima_lectura_ts - primera_lectura_ts
        duracion_calculada_segundos = duracion_delta.total_seconds()
    else:
        duracion_calculada_segundos = 0.0 # O None


    return SesionSummaryResponse(
        sesion_id=sesion_obj.sesion_id,
        trabajador_id=sesion_obj.trabajador_id,
        fecha_sesion=sesion_obj.fecha_sesion,
        hora_inicio=sesion_obj.hora_inicio,
        hora_fin=sesion_obj.hora_fin,
        estado_grabacion=sesion_obj.estado_grabacion,
        duracion_calculada_segundos=duracion_calculada_segundos,
        porcentaje_estres=porcentaje_estres,
        total_lecturas=total_lecturas
    )

# --- HELPER FUNCTION: Para calcular los detalles del resumen de una sesión ---
async def _calculate_session_summary_details(
    sesion_obj: Sesion, 
    db_session: AsyncSession
) -> Tuple[int, Optional[float], Optional[float]]:
    """
    Calcula total_lecturas, porcentaje_estres y duracion_calculada_segundos para una Sesion dada.
    """
    lecturas_statement = select(LecturaEstres).where(LecturaEstres.sesion_id == sesion_obj.sesion_id).order_by(LecturaEstres.timestamp_lectura)
    lecturas_result = await db_session.execute(lecturas_statement)
    lecturas = lecturas_result.scalars().all()
    
    total_lecturas = len(lecturas)
    porcentaje_estres: Optional[float] = None
    duracion_calculada_segundos: Optional[float] = None

    if total_lecturas > 0:
        conteo_estres = sum(1 for lec in lecturas if lec.prediccion == "Estrés")
        porcentaje_estres = (conteo_estres / total_lecturas) * 100
    else:
        porcentaje_estres = 0.0

    # Lógica de duración (como la definiste antes, puedes ajustarla si es necesario)
    # Prioridad 1: hora_fin y hora_inicio de la sesión
    if sesion_obj.estado_grabacion == "Finalizada" and sesion_obj.hora_fin and sesion_obj.hora_inicio:
        inicio_ts = sesion_obj.hora_inicio
        fin_ts = sesion_obj.hora_fin
        duracion_delta = fin_ts - inicio_ts
        duracion_calculada_segundos = duracion_delta.total_seconds()
    # Prioridad 2: Timestamps de lecturas
    elif total_lecturas > 0:
        primera_lectura_ts = lecturas[0].timestamp_lectura
        ultima_lectura_ts = lecturas[-1].timestamp_lectura
        duracion_delta = ultima_lectura_ts - primera_lectura_ts
        duracion_calculada_segundos = duracion_delta.total_seconds()
        if duracion_calculada_segundos < 0: # Pequeña salvaguarda
            duracion_calculada_segundos = 0.0
    else:
        duracion_calculada_segundos = 0.0
        
    return total_lecturas, porcentaje_estres, duracion_calculada_segundos

@app.get("/trabajadores/{trabajador_id}/sesiones/summary/", response_model=List[SesionSummaryResponse])
async def get_session_summaries_by_date_range(
    trabajador_id: int,
    start_date: Optional[date] = Query(None, description="Fecha de inicio del rango (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="Fecha de fin del rango (YYYY-MM-DD)"),
    db_session: AsyncSession = Depends(get_session)
):
    # --- Lógica de fechas (sin cambios) ---
    end_date_to_use = end_date or date.today()
    start_date_to_use = start_date or (end_date_to_use - timedelta(days=30))

    # --- Consulta de Sesiones (sin cambios) ---
    sesiones_statement = (
        select(Sesion)
        .where(
            Sesion.trabajador_id == trabajador_id,
            Sesion.fecha_sesion >= start_date_to_use,
            Sesion.fecha_sesion <= end_date_to_use,
            Sesion.estado_grabacion != 'Eliminado',
        )
        .order_by(Sesion.fecha_sesion.desc(), Sesion.hora_inicio.desc())
    )
    sesiones_results = await db_session.execute(sesiones_statement)
    sesiones_del_rango = sesiones_results.scalars().all()

    if not sesiones_del_rango:
        return []
    
    # --- [NUEVO] Consulta eficiente de Cuestionarios ---
    # 1. Obtenemos los IDs de todas las sesiones encontradas
    sesion_ids = [s.sesion_id for s in sesiones_del_rango if s.sesion_id is not None]

    cuestionarios_map: Dict[int, Cuestionario] = {}
    if sesion_ids:
        # 2. Hacemos UNA SOLA consulta para traer todos los cuestionarios de esas sesiones
        cuestionarios_stmt = select(Cuestionario).where(Cuestionario.sesion_id.in_(sesion_ids))
        cuestionarios_result = await db_session.execute(cuestionarios_stmt)
        # 3. Creamos un diccionario para búsqueda rápida (ID de sesión -> objeto Cuestionario)
        cuestionarios_map = {c.sesion_id: c for c in cuestionarios_result.scalars().all()}

    # --- Procesamiento y respuesta (modificado) ---
    summaries: List[SesionSummaryResponse] = []
    for sesion_obj in sesiones_del_rango:
        total_lecturas, porcentaje_estres, duracion_segundos = \
            await _calculate_session_summary_details(sesion_obj, db_session)
        
        # [NUEVO] Buscamos el cuestionario en nuestro mapa (sin llamar a la DB)
        cuestionario_asociado = cuestionarios_map.get(sesion_obj.sesion_id)
        
        summary = SesionSummaryResponse(
            sesion_id=sesion_obj.sesion_id,
            trabajador_id=sesion_obj.trabajador_id,
            fecha_sesion=sesion_obj.fecha_sesion,
            hora_inicio=sesion_obj.hora_inicio,
            hora_fin=sesion_obj.hora_fin,
            estado_grabacion=sesion_obj.estado_grabacion,
            duracion_calculada_segundos=duracion_segundos,
            porcentaje_estres=porcentaje_estres,
            total_lecturas=total_lecturas,
            cuestionario=cuestionario_asociado # <-- PASAMOS EL CUESTIONARIO
        )
        summaries.append(summary)

    return summaries


@app.get("/trabajadores/{trabajador_id}/", response_model=TrabajadorPublic, summary="Obtener un trabajador por su ID")
async def get_trabajador_by_id(
    trabajador_id: int,
    session: AsyncSession = Depends(get_session)
):
    """
    Obtiene y devuelve los datos públicos de un trabajador específico por su ID.
    """
    statement = select(Trabajador).where(Trabajador.trabajador_id == trabajador_id)
    result = await session.execute(statement)
    trabajador = result.scalars().first()

    if not trabajador:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trabajador no encontrado."
        )
    
    return trabajador


@app.put("/trabajadores/{trabajador_id}/", response_model=TrabajadorPublic, summary="Actualizar perfil de un trabajador")
async def update_trabajador_profile(
    trabajador_id: int,
    update_data: TrabajadorUpdate,
    session: AsyncSession = Depends(get_session)
):
    """
    Actualiza los datos del perfil de un trabajador específico.
    
    Este endpoint permite una actualización parcial. Solo los campos
    proporcionados en el cuerpo de la solicitud serán actualizados.
    El 'username' no puede ser modificado.
    """
    # 1. Buscar al trabajador en la base de datos por su ID
    statement = select(Trabajador).where(Trabajador.trabajador_id == trabajador_id)
    result = await session.execute(statement)
    db_trabajador = result.scalars().first()

    # Si no se encuentra el trabajador, devolver un error 404
    if not db_trabajador:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trabajador no encontrado."
        )

    # 2. Actualizar los campos del objeto trabajador
    # model_dump(exclude_unset=True) crea un diccionario solo con los
    # campos que el cliente envió, ignorando los que quedaron en None.
    update_data_dict = update_data.model_dump(exclude_unset=True)

    for key, value in update_data_dict.items():
        # setattr() actualiza el atributo del objeto dinámicamente
        setattr(db_trabajador, key, value)
        
    # 3. Actualizar la marca de tiempo de 'updated_at'
    db_trabajador.updated_at = datetime.now(timezone.utc)

    try:
        # 4. Guardar los cambios en la base de datos
        session.add(db_trabajador)
        await session.commit()
        await session.refresh(db_trabajador)
        return db_trabajador
    except Exception as e:
        await session.rollback()
        print(f"Error al actualizar el perfil del trabajador {trabajador_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ocurrió un error interno al actualizar el perfil."
        )


@app.get(
    "/trabajadores/{trabajador_id}/sesiones/summary/monthly/daily-aggregated/",
    response_model=List[DailyStressSummary], # Asegúrate que DailyStressSummary tenga los campos correctos
    summary="Resumen diario agregado (basado en todas las capturas del día) para un trabajador en un mes"
)
async def get_monthly_daily_aggregated_summary(
    trabajador_id: int,
    month: int = Query(..., ge=1, le=12, description="Número del mes (1-12)"),
    year: Optional[int] = Query(None, description="Año (ej. 2023). Si no se provee, se usa el año actual."),
    db_session: AsyncSession = Depends(get_session)
):
    current_dt = date.today()
    if year is None:
        year_to_filter = current_dt.year
    else:
        year_to_filter = year

    try:
        start_of_month = date(year_to_filter, month, 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Mes o año inválido.")

    if month == 12:
        start_of_next_month = date(year_to_filter + 1, 1, 1)
    else:
        start_of_next_month = date(year_to_filter, month + 1, 1)

    # 1. Obtener todas las sesiones del trabajador en el mes y año especificados
    sesiones_statement = select(Sesion).where(
        Sesion.trabajador_id == trabajador_id,
        Sesion.fecha_sesion >= start_of_month,
        Sesion.fecha_sesion < start_of_next_month,
        not_eliminado(Sesion.estado_grabacion),  # <-- filtro robusto

    ).order_by(Sesion.fecha_sesion)

    sesiones_results = await db_session.execute(sesiones_statement)
    sesiones_del_mes = sesiones_results.scalars().all()
    
    if not sesiones_del_mes:
        return []

    # 2. Agrupar datos y lecturas por día
    #    La clave será la fecha, y el valor un diccionario con los acumuladores para ese día
    daily_aggregated_data: Dict[date, Dict[str, any]] = {}

    for sesion_obj in sesiones_del_mes:
        
        estado_norm = (sesion_obj.estado_grabacion or "").strip().lower()
        if estado_norm == "eliminado":
            continue


        dia_sesion = sesion_obj.fecha_sesion

        # Inicializar el diccionario para el día si es la primera sesión de ese día
        if dia_sesion not in daily_aggregated_data:
            daily_aggregated_data[dia_sesion] = {
                "total_lecturas_estres_dia": 0,
                "total_lecturas_validas_dia": 0,
                "acum_duracion_segundos_dia": 0.0, # Para acumular duración
                "numero_sesiones_dia": 0
            }
        
        # a. Obtener todas las lecturas de ESTA sesión
        lecturas_de_esta_sesion_statement = select(LecturaEstres).where(
            LecturaEstres.sesion_id == sesion_obj.sesion_id
        ).order_by(LecturaEstres.timestamp_lectura) # Ordenar para cálculo de duración si es necesario

        lecturas_de_esta_sesion_result = await db_session.execute(lecturas_de_esta_sesion_statement)
        lecturas_de_esta_sesion_list = lecturas_de_esta_sesion_result.scalars().all()

        num_lecturas_esta_sesion = len(lecturas_de_esta_sesion_list)

        if num_lecturas_esta_sesion > 0:
            conteo_estres_esta_sesion = sum(1 for lec in lecturas_de_esta_sesion_list if lec.prediccion == "Estrés")
            
            # Acumular conteos para el DÍA
            daily_aggregated_data[dia_sesion]["total_lecturas_estres_dia"] += conteo_estres_esta_sesion
            daily_aggregated_data[dia_sesion]["total_lecturas_validas_dia"] += num_lecturas_esta_sesion

        # b. Calcular y acumular duración de esta sesión para el DÍA
        #    (Esta es la lógica de duración que tenías en _calculate_session_summary_details)
        duracion_segundos_esta_sesion: Optional[float] = None
        if sesion_obj.estado_grabacion == "Finalizada" and sesion_obj.hora_fin and sesion_obj.hora_inicio:
            # ... (lógica de duración basada en hora_inicio/fin de sesión) ...
            inicio_ts_s = sesion_obj.hora_inicio
            fin_ts_s = sesion_obj.hora_fin
            if inicio_ts_s and fin_ts_s:
                duracion_delta_s = fin_ts_s - inicio_ts_s
                duracion_segundos_esta_sesion = duracion_delta_s.total_seconds()
        elif num_lecturas_esta_sesion > 0:
            # ... (lógica de duración basada en primera/última lectura de la sesión) ...
            # lecturas_de_esta_sesion_list ya está ordenada
            primera_lectura_ts_s = lecturas_de_esta_sesion_list[0].timestamp_lectura
            ultima_lectura_ts_s = lecturas_de_esta_sesion_list[-1].timestamp_lectura
            if isinstance(primera_lectura_ts_s, datetime) and isinstance(ultima_lectura_ts_s, datetime):
                duracion_delta_s = ultima_lectura_ts_s - primera_lectura_ts_s
                duracion_segundos_esta_sesion = duracion_delta_s.total_seconds()
                if duracion_segundos_esta_sesion < 0: duracion_segundos_esta_sesion = 0.0
            else: # Manejar caso de timestamps no válidos
                duracion_segundos_esta_sesion = 0.0
        else:
            duracion_segundos_esta_sesion = 0.0
        
        if duracion_segundos_esta_sesion is not None:
            daily_aggregated_data[dia_sesion]["acum_duracion_segundos_dia"] += duracion_segundos_esta_sesion
        
        daily_aggregated_data[dia_sesion]["numero_sesiones_dia"] += 1


    # 3. Convertir los datos agrupados por día al formato de respuesta
    result_list: List[DailyStressSummary] = []
    sorted_days = sorted(daily_aggregated_data.keys()) # Ordenar los días para la respuesta

    for dia_fecha in sorted_days:
        data_del_dia = daily_aggregated_data[dia_fecha]
        
        porcentaje_estres_calculado_dia = None
        if data_del_dia["total_lecturas_validas_dia"] > 0:
            porcentaje_estres_calculado_dia = \
                (data_del_dia["total_lecturas_estres_dia"] / data_del_dia["total_lecturas_validas_dia"]) * 100
        
        result_list.append(
            DailyStressSummary(
                fecha=dia_fecha,
                porcentaje_estres_promedio=porcentaje_estres_calculado_dia, # Usar el valor agregado correcto
                duracion_total_grabacion_segundos=data_del_dia["acum_duracion_segundos_dia"],
                numero_sesiones=data_del_dia["numero_sesiones_dia"]
            )
        )
    
    return result_list


@app.get(
    "/trabajadores/{trabajador_id}/sesiones/summary/monthly/overall/",
    response_model=MonthlyOverallSummary,
    summary="Resumen general de estrés y grabación para un trabajador en un mes (basado en todas las capturas)"
)
async def get_monthly_overall_summary_for_worker(
    trabajador_id: int,
    month: int = Query(..., ge=1, le=12, description="Número del mes (1-12)"),
    year: Optional[int] = Query(None, description="Año (ej. 2023). Si no se provee, se usa el año actual."),
    db_session: AsyncSession = Depends(get_session)
):
    current_dt = date.today()
    if year is None:
        year_to_filter = current_dt.year
    else:
        year_to_filter = year

    try:
        start_of_month = date(year_to_filter, month, 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Mes o año inválido.")

    if month == 12:
        start_of_next_month = date(year_to_filter + 1, 1, 1)
    else:
        start_of_next_month = date(year_to_filter, month + 1, 1)

    # 1. Obtener todas las sesiones del trabajador en el mes y año especificados
    sesiones_statement = select(Sesion).where(
        Sesion.trabajador_id == trabajador_id,
        Sesion.fecha_sesion >= start_of_month,
        Sesion.fecha_sesion < start_of_next_month,
        not_eliminado(Sesion.estado_grabacion),  # <-- filtro robusto

    )
    sesiones_results = await db_session.execute(sesiones_statement)
    sesiones_del_mes = sesiones_results.scalars().all()

    if not sesiones_del_mes:
        return MonthlyOverallSummary(
            trabajador_id=trabajador_id,
            month=month,
            year=year_to_filter,
            tiempo_total_grabacion_mes_segundos=0.0,
            tiempo_promedio_grabacion_por_dia_activo_segundos=0.0,
            nivel_estres_promedio_mensual=None,
            numero_total_sesiones_mes=0,
            promedio_sesiones_por_dia_activo=0.0,
            dias_con_actividad=0
        )

    # Inicializar acumuladores para las métricas del mes
    total_tiempo_grabacion_mes_segundos_acum = 0.0
    total_lecturas_estres_mes_acum = 0
    total_lecturas_validas_mes_acum = 0 # Suma de todas las lecturas de todas las sesiones del mes

    numero_total_sesiones_mes = len(sesiones_del_mes)
    dias_activos_set = set() # Para contar días únicos con actividad

    # 2. Iterar sobre cada sesión del mes para acumular datos
    for sesion_obj in sesiones_del_mes:
        if (sesion_obj.estado_grabacion or "").strip().lower() == "eliminado":
            continue

        dias_activos_set.add(sesion_obj.fecha_sesion)
        
        # a. Acumular conteos de estrés y lecturas totales para esta sesión
        lecturas_sesion_statement = select(LecturaEstres).where(LecturaEstres.sesion_id == sesion_obj.sesion_id)
        lecturas_sesion_result = await db_session.execute(lecturas_sesion_statement)
        lecturas_de_la_sesion = lecturas_sesion_result.scalars().all()

        num_lecturas_esta_sesion = len(lecturas_de_la_sesion)
        
        if num_lecturas_esta_sesion > 0:
            conteo_estres_esta_sesion = sum(1 for lec in lecturas_de_la_sesion if lec.prediccion == "Estrés")
            total_lecturas_estres_mes_acum += conteo_estres_esta_sesion
            total_lecturas_validas_mes_acum += num_lecturas_esta_sesion

        # b. Calcular y acumular duración de esta sesión
        #    (Esta es la lógica de duración que tenías en _calculate_session_summary_details)
        duracion_segundos_esta_sesion: Optional[float] = None
        if sesion_obj.estado_grabacion == "Finalizada" and sesion_obj.hora_fin and sesion_obj.hora_inicio:
            inicio_ts_s = sesion_obj.hora_inicio
            fin_ts_s = sesion_obj.hora_fin
            # Asegurarse de que ambos son conscientes de la zona horaria o ambos son naive
            if inicio_ts_s and fin_ts_s: # Pequeña comprobación adicional
                 # Asumiendo que están correctamente configurados con timezone en la DB
                duracion_delta_s = fin_ts_s - inicio_ts_s
                duracion_segundos_esta_sesion = duracion_delta_s.total_seconds()
        elif num_lecturas_esta_sesion > 0: # Usar lecturas si no está finalizada o faltan horas
            # Asegurarse de que las lecturas estén ordenadas si se usa este método
            # Para este ejemplo, asumimos que se obtuvieron en orden o se reordenarían aquí si fuera necesario
            # Para mayor precisión, se debe asegurar el orden por timestamp_lectura
            # Re-consultar ordenado o ordenar 'lecturas_de_la_sesion'
            if not all(isinstance(l.timestamp_lectura, datetime) for l in lecturas_de_la_sesion):
                 # Manejar caso donde los timestamps no son válidos (poco probable con type hints)
                pass # O loguear un warning
            else:
                # Ordenar por si acaso, aunque la consulta original para _calculate_session_summary_details lo hacía
                lecturas_ordenadas_para_duracion = sorted(lecturas_de_la_sesion, key=lambda l: l.timestamp_lectura)
                primera_lectura_ts_s = lecturas_ordenadas_para_duracion[0].timestamp_lectura
                ultima_lectura_ts_s = lecturas_ordenadas_para_duracion[-1].timestamp_lectura
                duracion_delta_s = ultima_lectura_ts_s - primera_lectura_ts_s
                duracion_segundos_esta_sesion = duracion_delta_s.total_seconds()
                if duracion_segundos_esta_sesion < 0:
                    duracion_segundos_esta_sesion = 0.0
        else: # No hay lecturas y no está finalizada con horas válidas
            duracion_segundos_esta_sesion = 0.0
        
        if duracion_segundos_esta_sesion is not None:
            total_tiempo_grabacion_mes_segundos_acum += duracion_segundos_esta_sesion
            
    # 3. Calcular las métricas finales del mes
    dias_con_actividad_count = len(dias_activos_set)
    
    nivel_estres_promedio_mensual_final = None
    if total_lecturas_validas_mes_acum > 0:
        nivel_estres_promedio_mensual_final = \
            (total_lecturas_estres_mes_acum / total_lecturas_validas_mes_acum) * 100

    tiempo_promedio_grabacion_dia_activo_final = None
    if dias_con_actividad_count > 0 and total_tiempo_grabacion_mes_segundos_acum > 0: # Evitar división por cero y asegurar que hubo grabación
        tiempo_promedio_grabacion_dia_activo_final = \
            total_tiempo_grabacion_mes_segundos_acum / dias_con_actividad_count

    promedio_sesiones_dia_activo_final = None
    if dias_con_actividad_count > 0:
        promedio_sesiones_dia_activo_final = \
            numero_total_sesiones_mes / dias_con_actividad_count
        
    return MonthlyOverallSummary(
        trabajador_id=trabajador_id,
        month=month,
        year=year_to_filter,
        tiempo_total_grabacion_mes_segundos=total_tiempo_grabacion_mes_segundos_acum,
        tiempo_promedio_grabacion_por_dia_activo_segundos=tiempo_promedio_grabacion_dia_activo_final,
        nivel_estres_promedio_mensual=nivel_estres_promedio_mensual_final,
        numero_total_sesiones_mes=numero_total_sesiones_mes,
        promedio_sesiones_por_dia_activo=promedio_sesiones_dia_activo_final,
        dias_con_actividad=dias_con_actividad_count
    )

@app.get("/cuestionarios/", response_model=List[Cuestionario], summary="Obtener todos los cuestionarios")
async def get_all_cuestionarios(
    session: AsyncSession = Depends(get_session)
):
    """
    Obtiene y devuelve una lista de todos los cuestionarios registrados en la base de datos.
    """
    try:
        # Crea una sentencia para seleccionar todos los registros de la tabla 'cuestionarios'
        statement = select(Cuestionario).order_by(Cuestionario.created_at.desc()) # Opcional: ordenar por más reciente
        
        # Ejecuta la sentencia de forma asíncrona
        results = await session.execute(statement)
        
        # Obtiene todos los objetos Cuestionario
        cuestionarios = results.scalars().all()
        
        return cuestionarios
        
    except Exception as e:
        # Manejo de errores genérico
        print(f"Error al obtener cuestionarios: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Ocurrió un error al intentar obtener la lista de cuestionarios."
        )

# Filtro robusto para excluir sesiones "Eliminado"
def not_eliminado(expr):
    # expr debe ser la columna Sesion.estado_grabacion
    return func.lower(func.trim(func.coalesce(expr, ""))) != "eliminado"


@app.post(
    "/cuestionarios/", 
    response_model=Cuestionario, 
    status_code=status.HTTP_201_CREATED,
    summary="Crear un nuevo cuestionario para una sesión"
)
async def create_cuestionario(
    cuestionario_data: CuestionarioCreate,
    session: AsyncSession = Depends(get_session)
):
    """
    Crea un nuevo registro de cuestionario asociado a una sesión.
    Valida que la sesión exista y que no tenga ya un cuestionario.
    """
    # 1. Validar que la sesión exista
    sesion_stmt = select(Sesion).where(Sesion.sesion_id == cuestionario_data.sesion_id)
    sesion_result = await session.execute(sesion_stmt)
    if not sesion_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"La sesión con ID {cuestionario_data.sesion_id} no fue encontrada."
        )

    # 2. Validar que no exista ya un cuestionario para esta sesión (regla de 1 a 1)
    existing_q_stmt = select(Cuestionario).where(Cuestionario.sesion_id == cuestionario_data.sesion_id)
    existing_q_result = await session.execute(existing_q_stmt)
    if existing_q_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ya existe un cuestionario registrado para la sesión ID {cuestionario_data.sesion_id}."
        )
        
    # 3. Crear el nuevo objeto Cuestionario
    current_utc_time = datetime.now(timezone.utc)
    new_cuestionario = Cuestionario(
        **cuestionario_data.model_dump(), # Desempaqueta los datos del modelo Pydantic
        created_at=current_utc_time,
        updated_at=current_utc_time
    )

    try:
        session.add(new_cuestionario)
        await session.commit()
        await session.refresh(new_cuestionario)
        return new_cuestionario
    except Exception as e:
        await session.rollback() # Importante hacer rollback en caso de error
        print(f"Error al crear el cuestionario: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al guardar el cuestionario."
        )



@app.get("/cuestionarios/", response_model=List[Cuestionario], summary="Obtener todos los cuestionarios")
async def get_all_cuestionarios(
    session: AsyncSession = Depends(get_session)
):
    """
    Obtiene y devuelve una lista de todos los cuestionarios registrados en la base de datos.
    """
    try:
        # Crea una sentencia para seleccionar todos los registros de la tabla 'cuestionarios'
        statement = select(Cuestionario).order_by(Cuestionario.created_at.desc()) # Opcional: ordenar por más reciente
        
        # Ejecuta la sentencia de forma asíncrona
        results = await session.execute(statement)
        
        # Obtiene todos los objetos Cuestionario
        cuestionarios = results.scalars().all()
        
        return cuestionarios
        
    except Exception as e:
        # Manejo de errores genérico
        print(f"Error al obtener cuestionarios: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Ocurrió un error al intentar obtener la lista de cuestionarios."
        )

@app.delete("/sessions/{sesion_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    sesion_id: int = Path(..., description="ID de la sesión a eliminar"),
    session: AsyncSession = Depends(get_session)
):
    """
    Elimina una sesión y sus datos asociados (lecturas y cuestionario).
    Si tu BD ya tiene ON DELETE CASCADE, puedes omitir los deletes de dependientes.
    """
    try:
        # 1) Verificar existencia
        stmt = select(Sesion).where(Sesion.sesion_id == sesion_id)
        res = await session.execute(stmt)
        sesion_obj = res.scalars().first()
        if not sesion_obj:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND,
                                detail=f"La sesión con ID {sesion_id} no fue encontrada.")

        # 2) Borrar dependientes (si no hay CASCADE en la BD)
        await session.execute(delete(LecturaEstres).where(LecturaEstres.sesion_id == sesion_id))
        await session.execute(delete(Cuestionario).where(Cuestionario.sesion_id == sesion_id))

        # 3) Borrar la sesión
        await session.delete(sesion_obj)
        await session.commit()

        # 204 No Content (sin body)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        print(f"Error al eliminar la sesión {sesion_id}: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                            detail="Error interno al intentar eliminar la sesión.")


@app.post(
    "/cuestionarios/", 
    response_model=Cuestionario, 
    status_code=status.HTTP_201_CREATED,
    summary="Crear un nuevo cuestionario para una sesión"
)
async def create_cuestionario(
    cuestionario_data: CuestionarioCreate,
    session: AsyncSession = Depends(get_session)
):
    """
    Crea un nuevo registro de cuestionario asociado a una sesión.
    Valida que la sesión exista y que no tenga ya un cuestionario.
    """
    # 1. Validar que la sesión exista
    sesion_stmt = select(Sesion).where(Sesion.sesion_id == cuestionario_data.sesion_id)
    sesion_result = await session.execute(sesion_stmt)
    if not sesion_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"La sesión con ID {cuestionario_data.sesion_id} no fue encontrada."
        )

    # 2. Validar que no exista ya un cuestionario para esta sesión (regla de 1 a 1)
    existing_q_stmt = select(Cuestionario).where(Cuestionario.sesion_id == cuestionario_data.sesion_id)
    existing_q_result = await session.execute(existing_q_stmt)
    if existing_q_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ya existe un cuestionario registrado para la sesión ID {cuestionario_data.sesion_id}."
        )
        
    # 3. Crear el nuevo objeto Cuestionario
    current_utc_time = datetime.now(timezone.utc)
    new_cuestionario = Cuestionario(
        **cuestionario_data.model_dump(), # Desempaqueta los datos del modelo Pydantic
        created_at=current_utc_time,
        updated_at=current_utc_time
    )

    try:
        session.add(new_cuestionario)
        await session.commit()
        await session.refresh(new_cuestionario)
        return new_cuestionario
    except Exception as e:
        await session.rollback() # Importante hacer rollback en caso de error
        print(f"Error al crear el cuestionario: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al guardar el cuestionario."
        )

@app.get("/trabajadores/{trabajador_id}", response_model=TrabajadorBasic)
async def get_trabajador_by_id(
    trabajador_id: int,
    db: AsyncSession = Depends(get_session)
):
    stmt = select(Trabajador).where(Trabajador.trabajador_id == trabajador_id)
    res = await db.execute(stmt)
    t = res.scalars().first()
    if not t:
        raise HTTPException(status_code=404, detail="Trabajador no encontrado")
    return TrabajadorBasic(
        trabajador_id=t.trabajador_id,
        nombre=t.nombre,
        username=t.username,
        role_id=getattr(t, "role_id", None)  # asumiendo que ya agregaste role_id en tu tabla/modelo
    )


@app.get("/jefes/{jefe_id}/subordinados/", response_model=List[TrabajadorBasic])
async def get_subordinados_de_jefe(
    jefe_id: int,
    db: AsyncSession = Depends(get_session)
):
    sql = text("""
        SELECT t.trabajador_id, t.nombre, t.username, t.role_id
        FROM trabajador_jefe tj
        JOIN trabajadores t ON t.trabajador_id = tj.trabajador_id
        WHERE tj.jefe_id = :jefe_id
        ORDER BY t.nombre
    """)
    res = await db.execute(sql, {"jefe_id": jefe_id})
    rows = res.mappings().all()
    return [
        TrabajadorBasic(
            trabajador_id=row["trabajador_id"],
            nombre=row["nombre"],
            username=row["username"],
            role_id=row["role_id"]
        ) for row in rows
    ]



@app.get("/trabajadores/{trabajador_id}/basic", response_model=TrabajadorBasic)
async def get_trabajador_basic_by_id(
    trabajador_id: int,
    db: AsyncSession = Depends(get_session)
):
    stmt = select(Trabajador).where(Trabajador.trabajador_id == trabajador_id)
    res = await db.execute(stmt)
    t = res.scalars().first()
    if not t:
        raise HTTPException(status_code=404, detail="Trabajador no encontrado")
    return TrabajadorBasic(
        trabajador_id=t.trabajador_id,
        nombre=t.nombre,
        username=t.username,
        role_id=getattr(t, "role_id", None)
    )

@app.get("/jefes/{jefe_id}/subordinados/", response_model=List[TrabajadorBasic])
async def get_subordinados_de_jefe(
    jefe_id: int,
    db: AsyncSession = Depends(get_session)
):
    sql = text("""
        SELECT t.trabajador_id, t.nombre, t.username, t.role_id
        FROM trabajador_jefe tj
        JOIN trabajadores t ON t.trabajador_id = tj.trabajador_id
        WHERE tj.jefe_id = :jefe_id
        ORDER BY t.nombre
    """)
    res = await db.execute(sql, {"jefe_id": jefe_id})
    rows = res.mappings().all()
    return [
        TrabajadorBasic(
            trabajador_id=row["trabajador_id"],
            nombre=row["nombre"],
            username=row["username"],
            role_id=row["role_id"]
        ) for row in rows
    ]

@app.get("/admin/usuarios")
async def admin_listar_usuarios(
    role_id: Optional[int] = Query(None),
    session: AsyncSession = Depends(get_session)
):
    stmt = select(Trabajador)
    if role_id is not None:
        stmt = stmt.where(Trabajador.role_id == role_id)
    res = await session.execute(stmt)
    return res.scalars().all()


@app.get("/admin/jefes")
async def admin_listar_jefes(session: AsyncSession = Depends(get_session)):
    stmt = select(Trabajador).where(Trabajador.role_id == 3)
    res = await session.execute(stmt)
    return res.scalars().all()

@app.get("/admin/usuarios/sin-jefe")
async def admin_usuarios_sin_jefe(session: AsyncSession = Depends(get_session)):
    sql = text("""
      SELECT t.trabajador_id, t.nombre, t.username, t.role_id
      FROM trabajadores t
      WHERE t.trabajador_id NOT IN (SELECT trabajador_id FROM trabajador_jefe)
      ORDER BY t.nombre
    """)
    res = await session.execute(sql)
    return res.mappings().all()


@app.put("/admin/usuarios/{trabajador_id}/rol")
async def admin_actualizar_rol(
    trabajador_id: int,
    payload: AdminRoleUpdate,
    session: AsyncSession = Depends(get_session)
):
    q = select(Trabajador).where(Trabajador.trabajador_id == trabajador_id)
    res = await session.execute(q)
    t = res.scalars().first()
    if not t:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    t.role_id = payload.role_id
    t.updated_at = datetime.now(timezone.utc)

    await session.commit()
    await session.refresh(t)
    msg = "Rol asignado con éxito." if payload.role_id == 3 else "Rol modificado con éxito."
    return {"message": msg, "trabajador": t}



@app.delete("/admin/usuarios/{trabajador_id}")
async def admin_eliminar_usuario(
    trabajador_id: int,
    session: AsyncSession = Depends(get_session)
):
    # 1) verificar
    res = await session.execute(select(Trabajador).where(Trabajador.trabajador_id == trabajador_id))
    user = res.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    # 2) borrar relaciones (si no hay CASCADE)
    await session.execute(delete(TrabajadorJefe).where(
        (TrabajadorJefe.trabajador_id == trabajador_id) | (TrabajadorJefe.jefe_id == trabajador_id)
    ))
    # borra sesiones y dependientes
    sesiones_res = await session.execute(select(Sesion).where(Sesion.trabajador_id == trabajador_id))
    sesiones = sesiones_res.scalars().all()
    for s in sesiones:
        await session.execute(delete(LecturaEstres).where(LecturaEstres.sesion_id == s.sesion_id))
        await session.execute(delete(Cuestionario).where(Cuestionario.sesion_id == s.sesion_id))
        await session.delete(s)

    # 3) borrar usuario
    await session.delete(user)
    await session.commit()
    return {"message": "Usuario eliminado exitosamente."}

@app.delete("/admin/asignaciones")
async def admin_eliminar_asignacion(
    payload: AsignacionDelete,
    session: AsyncSession = Depends(get_session)
):
    # existe?
    q = select(TrabajadorJefe).where(
        (TrabajadorJefe.jefe_id == payload.jefe_id) &
        (TrabajadorJefe.trabajador_id == payload.subordinado_id)
    )
    res = await session.execute(q)
    rel = res.scalars().first()
    if not rel:
        raise HTTPException(status_code=404, detail="Relación no encontrada")
    await session.delete(rel)
    await session.commit()
    return {"message": "Asignación eliminada correctamente."}


@app.post("/admin/asignaciones")
async def admin_asignar_subordinados(
    payload: AsignacionCreate,
    session: AsyncSession = Depends(get_session)
):
    # verifica que el jefe sea rol 3
    qj = select(Trabajador).where(Trabajador.trabajador_id == payload.jefe_id)
    rj = await session.execute(qj)
    jefe = rj.scalars().first()
    if not jefe or jefe.role_id != 3:
        raise HTTPException(status_code=400, detail="El jefe no es válido o no tiene rol de jefe (3).")

    # verifica subordinados libres y no asignados
    for sub_id in payload.subordinado_ids:
        # ya asignado?
        qs = select(TrabajadorJefe).where(TrabajadorJefe.trabajador_id == sub_id)
        rs = await session.execute(qs)
        if rs.scalars().first():
            # CP054
            raise HTTPException(status_code=409, detail=f"No se puede realizar la asignación: colaborador {sub_id} ya vinculado.")

    # persiste
    for sub_id in payload.subordinado_ids:
        session.add(TrabajadorJefe(jefe_id=payload.jefe_id, trabajador_id=sub_id))
    await session.commit()

    return {"message": "Asignación registrada exitosamente."}



# Desplegar la aplicacion usando uvicorn como servidor ASGI
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
