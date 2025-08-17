# backend/main.py
import asyncio
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple
import bcrypt
from fastapi import FastAPI, File, Form, Path, Query, UploadFile, HTTPException, Depends, status # Importar Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from database import create_db_and_tables, get_session # Importar funciones de database.py
from sqlalchemy.ext.asyncio import AsyncSession # Importar AsyncSession
from sqlalchemy.exc import IntegrityError
import logging
from anyio import to_thread

from sqlmodel import Session, select
import torch
from torchvision import transforms # Seguiremos usando esto para el preprocesamiento
from PIL import Image
from io import BytesIO
import os
import sys

from models import Trabajador, LecturaEstres, Sesion, Cuestionario

# QUITAR IMPORTACIONES DE FASTAI SI SOLO CARGAMOS UN nn.Module
# from fastai.vision.all import load_learner, PILImage # << YA NO LAS NECESITAMOS (probablemente)

# --- CONFIGURACIÓN ---
MODEL_DIR = os.path.dirname(__file__)
# Corregir el path para que no tenga "./" extra, aunque debería funcionar igual
MODEL_PATH = os.path.join(MODEL_DIR, "stress.pth")
# O más simple si stress.pth está junto a main.py:
# MODEL_PATH = "stress.pth" # O os.path.join(os.path.dirname(__file__), "stress.pth")

CLASSES = ["No Estrés", "Estrés"]

# Las transformaciones de torchvision ahora serán CLAVE
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Asegúrate que esto coincida con el entrenamiento
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Y esto también
])


class UserCreate(BaseModel):
    nombre: str
    username: str
    password: str
    fecha_de_nacimiento: Optional[date] = None
    genero: Optional[str] = None
    estado_civil: Optional[str] = None
    uso_de_anteojos: Optional[bool] = False
    estudio_y_trabajo: Optional[str] = None
    horas_trabajo_semanal: Optional[int] = None
    horas_descanso_dia: Optional[int] = None
    
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
    cuestionario: Optional[Cuestionario] = None # <-- CAMPO AÑADIDO


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


# --- CARGAR EL MODELO (AHORA ASUMIENDO UN nn.Module DE PYTORCH) ---
model = None # 'model' será el nn.Module
try:
    print(f"Intentando cargar el modelo PyTorch (nn.Module) desde: {MODEL_PATH}")
    # map_location para asegurar que se carga en CPU.
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)

    # Verificar si el objeto cargado es realmente un nn.Module
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"El archivo cargado no es un nn.Module. Tipo encontrado: {type(model)}")

    model.eval() # Poner el modelo en modo evaluación
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


def _infer_sync(model, image_tensor, CLASSES):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        idx = int(torch.argmax(probabilities).item())
        return CLASSES[idx], float(probabilities[idx].item())


# --- CONFIGURAR FastAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        await commit_with_retry(session)
        await session.refresh(new_trabajador)
        return new_trabajador
    except IntegrityError:
        # Este bloque se ejecutará si la restricción 'unique=True' del username falla.
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="El nombre de usuario ya existe. Por favor, elige otro."
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


@app.post("/predict/")
async def predict_stress(
    file: UploadFile = File(...),
    sesion_id: int = Form(...),
    session: AsyncSession = Depends(get_session)
):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo de inferencia no está disponible.")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo subido debe ser una imagen.")

    # 1) leer y preparar imagen
    contents = await file.read()
    try:
        image_pil = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo abrir o procesar la imagen: {e}")

    image_tensor = transform(image_pil).unsqueeze(0)
    device = next(model.parameters()).device  # CPU
    image_tensor = image_tensor.to(device)

    # 2) inferencia en hilo (no bloquea el event loop)
    predicted_class_name, confidence = await to_thread.run_sync(
        _infer_sync, model, image_tensor, CLASSES
    )

    # 3) guardar en DB con retry
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
        await commit_with_retry(session)
        await session.refresh(lectura_individual)

    except Exception as e:
        # log opcional aquí
        raise HTTPException(status_code=500, detail=f"Error durante el procesamiento o guardado: {e}")

    return JSONResponse(content={
        "prediction": predicted_class_name,
        "confidence": round(confidence, 4),
        "lectura_estres_id": lectura_individual.lectura_estres_id
    })


async def commit_with_retry(session, attempts=3):
    backoff = 0.5
    for i in range(attempts):
        try:
            await session.commit()
            return
        except (asyncio.TimeoutError, Exception):
            if i == attempts - 1:
                await session.rollback()
                raise
            await asyncio.sleep(backoff)
            backoff *= 2



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
        await commit_with_retry(session)
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
        await commit_with_retry(session)
        await session.refresh(db_session)
        return db_session
    except Exception as e:
        print(f"Error al finalizar sesión {sesion_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error al finalizar la sesión de grabación.")

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
            Sesion.fecha_sesion <= end_date_to_use
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
        Sesion.fecha_sesion < start_of_next_month
    ).order_by(Sesion.fecha_sesion)

    sesiones_results = await db_session.execute(sesiones_statement)
    sesiones_del_mes = sesiones_results.scalars().all()
    
    if not sesiones_del_mes:
        return []

    # 2. Agrupar datos y lecturas por día
    #    La clave será la fecha, y el valor un diccionario con los acumuladores para ese día
    daily_aggregated_data: Dict[date, Dict[str, any]] = {}

    for sesion_obj in sesiones_del_mes:
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
        Sesion.fecha_sesion < start_of_next_month
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
        await commit_with_retry(session)
        await session.refresh(new_cuestionario)
        return new_cuestionario
    except Exception as e:
        await session.rollback() # Importante hacer rollback en caso de error
        print(f"Error al crear el cuestionario: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al guardar el cuestionario."
        )




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)