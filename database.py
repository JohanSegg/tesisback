# backend/database.py
import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Carga .env en local (en Render no hace falta; Render ya inyecta env vars)
load_dotenv()

# Lee la URL desde env (con fallback local)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:root@localhost:5433/postgres"
)

# NOTA SSL:
# - Si conectas a un Postgres externo que exige SSL con asyncpg,
#   agrega '?ssl=true' a la DATABASE_URL (recomendado)
#   ej: postgresql+asyncpg://user:pass@host:5432/db?ssl=true

# Crea engine asíncrono
engine = create_async_engine(
    DATABASE_URL,
    echo=True,                # pon False en prod si quieres menos logs
    pool_pre_ping=True        # evita conexiones muertas
    # connect_args={}         # rara vez necesario con asyncpg si usas ?ssl=true
)

# Fábrica de sesiones
async_session_factory = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def create_db_and_tables():
    print("Intentando crear tablas en la base de datos...")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    print("Tablas verificadas/creadas en la base de datos.")

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
