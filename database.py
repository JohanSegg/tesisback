# backend/database.py
from sqlmodel import SQLModel # Solo necesitamos SQLModel para el metadata
import os
from typing import AsyncGenerator

# Importar AsyncSession y create_async_engine para el manejo de motores asíncronos
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Configuración de la URL de la base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:root@localhost:5433/postgres")

# Crear el motor de la base de datos asíncrono
engine = create_async_engine(DATABASE_URL, echo=True)

# Crear una fábrica de sesiones asíncronas
# Aquí especificamos explícitamente AsyncSession de SQLAlchemy.
# Esto asegura que la sesión devuelta por la fábrica sea compatible con 'async with'.
async_session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Función para crear todas las tablas definidas en los modelos
async def create_db_and_tables():
    print("Intentando crear tablas en la base de datos...")
    async with engine.begin() as conn:
        # run_sync se usa para ejecutar operaciones síncronas de SQLAlchemy
        # dentro de un contexto asíncrono, como la creación de tablas.
        await conn.run_sync(SQLModel.metadata.create_all)
    print("Tablas verificadas/creadas en la base de datos.")

# Dependencia para obtener una sesión de base de datos asíncrona
# El tipo de retorno es AsyncSession, que es lo que FastAPI inyectará.
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
