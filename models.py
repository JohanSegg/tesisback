# models.py
from typing import Optional
from sqlmodel import Column, DateTime, Field, SQLModel
from datetime import date, datetime, timezone

# --- Tabla trabajadores ---
class Trabajador(SQLModel, table=True):

    __tablename__ = "trabajadores"

    # Clave primaria autoincremental
    trabajador_id: Optional[int] = Field(default=None, primary_key=True)

    # Campos
    nombre: str = Field(max_length=255, nullable=False)
    username: str = Field(max_length=100, nullable=False, unique=True)
    password: str = Field(max_length=255, nullable=False)
    fecha_de_nacimiento: Optional[date] = Field(default=None)
    genero: Optional[str] = Field(max_length=15, default=None)
    estado_civil: Optional[str] = Field(max_length=20, default=None)
    uso_de_anteojos: Optional[bool] = Field(default=False)
    estudio_y_trabajo: Optional[str] = Field(max_length=255, default=None)
    horas_trabajo_semanal: Optional[int] = Field(default=None)
    horas_descanso_dia: Optional[int] = Field(default=None)

    # created_at y updated_at se inicializan con la hora UTC actual.
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) 
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) 
    )


# --- Tabla sesiones ---
class Sesion(SQLModel, table=True):
    __tablename__ = "sesiones"

    # Clave primaria autoincremental
    sesion_id: Optional[int] = Field(default=None, primary_key=True)

    # Campos
    trabajador_id: int = Field(foreign_key="trabajadores.trabajador_id") # llave foranea 
    fecha_sesion: date
    hora_inicio: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) 
    )
    hora_fin: Optional[datetime] = Field(
        default=None, 
        sa_column=Column(DateTime(timezone=True), nullable=True) 
    )
    estado_grabacion: Optional[str] = Field(max_length=50, default=None)
    duracion_grabacion_segundos: Optional[int] = Field(default=None)
    nivel_estres_modelo_avg: Optional[float] = Field(default=None)
    distancia_rostro_pantalla_avg: Optional[float] = Field(default=None)
    brillo_pantalla_avg: Optional[float] = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) 
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) 
    )

# --- Tabla cuestionarios ---
class Cuestionario(SQLModel, table=True):
    __tablename__ = "cuestionarios"

    # Clave primaria autoincremental
    cuestionario_id: Optional[int] = Field(default=None, primary_key=True)

    # Campos
    sesion_id: int = Field(foreign_key="sesiones.sesion_id") # llave foranea 
    descripcion_trabajo: Optional[str] = Field(default=None)
    nivel_de_sensacion_estres: Optional[int] = Field(default=None)
    molestias_fisicas_visual: Optional[int] = Field(default=None)
    molestias_fisicas_otros: Optional[int] = Field(default=None)
    dificultad_concentracion: Optional[int] = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) 
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) 
    )

# --- Tabla lecturas_estres ---
class LecturaEstres(SQLModel, table=True):
    __tablename__ = "lecturas_estres"

    #id
    lectura_estres_id: Optional[int] = Field(default=None, primary_key=True)


    #Campos
    sesion_id: int = Field(foreign_key="sesiones.sesion_id") #  otra llave foranea!
    prediccion: str = Field(max_length=50, nullable=False)
    timestamp_lectura: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) 
    )
    confianza: float = Field(nullable=False)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) 
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False) 
    )

