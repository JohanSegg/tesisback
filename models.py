# backend/models.py
from typing import Optional
from sqlmodel import Column, DateTime, Field, SQLModel
from datetime import date, datetime, timezone

# Modelo para la tabla 'trabajadores'
# Los nombres de los campos y tipos de datos deben coincidir con tu script SQL.
class Trabajador(SQLModel, table=True):
    # Nombre de la tabla en la base de datos (en plural y snake_case)
    __tablename__ = "trabajadores"

    # Clave primaria autoincremental
    trabajador_id: Optional[int] = Field(default=None, primary_key=True)

    # Campos de la tabla 'trabajadores'
    nombre: str = Field(max_length=255, nullable=False)
    username: str = Field(max_length=100, nullable=False, unique=True)
    correo: str = Field(max_length=100, nullable=False, unique=True)
    password: str = Field(max_length=255, nullable=False)
    fecha_de_nacimiento: Optional[date] = Field(default=None)
    genero: Optional[str] = Field(max_length=15, default=None)
    estado_civil: Optional[str] = Field(max_length=20, default=None)
    uso_de_anteojos: Optional[bool] = Field(default=False)
    estudio_y_trabajo: Optional[str] = Field(max_length=255, default=None)
    horas_trabajo_semanal: Optional[int] = Field(default=None)
    horas_descanso_dia: Optional[int] = Field(default=None)

    # created_at y updated_at (gestionados manualmente desde el backend)
    # Se inicializan con la hora UTC actual.
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) # <-- Nuevo mapeo
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) # <-- Nuevo mapeo
    )

# Puedes añadir aquí otros modelos como Sesion, Cuestionario, LecturaEstres
# Por ejemplo, para la tabla 'sesiones':
class Sesion(SQLModel, table=True):
    __tablename__ = "sesiones"
    sesion_id: Optional[int] = Field(default=None, primary_key=True)
    trabajador_id: int = Field(foreign_key="trabajadores.trabajador_id")
    fecha_sesion: date
    hora_inicio: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) # <-- Nuevo mapeo
    )
    hora_fin: Optional[datetime] = Field(
        default=None, # O simplemente omite el default_factory
        sa_column=Column(DateTime(timezone=True), nullable=True) # <-- Nuevo mapeo
    )
    estado_grabacion: Optional[str] = Field(max_length=50, default=None)
    duracion_grabacion_segundos: Optional[int] = Field(default=None)
    nivel_estres_modelo_avg: Optional[float] = Field(default=None)
    distancia_rostro_pantalla_avg: Optional[float] = Field(default=None)
    brillo_pantalla_avg: Optional[float] = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) # <-- Nuevo mapeo
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) # <-- Nuevo mapeo
    )

# Modelo para la tabla 'cuestionarios'
class Cuestionario(SQLModel, table=True):
    __tablename__ = "cuestionarios"
    cuestionario_id: Optional[int] = Field(default=None, primary_key=True)
    sesion_id: int = Field(foreign_key="sesiones.sesion_id")
    descripcion_trabajo: Optional[str] = Field(default=None)
    nivel_de_sensacion_estres: Optional[int] = Field(default=None)
    molestias_fisicas_visual: Optional[int] = Field(default=None)
    molestias_fisicas_otros: Optional[int] = Field(default=None)
    dificultad_concentracion: Optional[int] = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) # <-- Nuevo mapeo
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) # <-- Nuevo mapeo
    )

# Modelo para la tabla 'lecturas_estres'
class LecturaEstres(SQLModel, table=True):
    __tablename__ = "lecturas_estres"
    lectura_estres_id: Optional[int] = Field(default=None, primary_key=True)
    sesion_id: int = Field(foreign_key="sesiones.sesion_id")
    prediccion: str = Field(max_length=50, nullable=False)
    timestamp_lectura: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) # <-- Nuevo mapeo
    )
    confianza: float = Field(nullable=False)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) # <-- Nuevo mapeo
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) # <-- Nuevo mapeo
    )
