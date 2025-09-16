# database.py
import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Carga .env en local
load_dotenv()

# URL de la BD
DATABASE_URL = os.getenv("DATABASE_URL", "")
# Usar fallback local (lo que esta despues de la coma) si no funciona: 
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://usuario:contraseña@host:5432/basededatos") 
# NOTA:
# Si hay problemas con ssl  recomendado agregar "?ssl=true" al final de DATABASE_URL 


# Funcion que crea un motor de conexión asíncrono
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,          
    pool_recycle=300,            # Recicla conexiones cada 300 s (5 min)
    pool_size=5,              
    max_overflow=10,           
    connect_args={
        # Para asyncpg:
        "timeout": 30,           
        "command_timeout": 60,   
        "ssl": "require",       # Fuerza SSL con asyncpg. Si falla, cambiar por True
        # "statement_cache_size": 0,  #  desactiva cache
    },
)

async_session_factory = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Crear la base de datos y verificarla
async def create_db_and_tables():
    print("Intentando crear tablas en la base de datos...")
    async with engine.begin() as conn:
        # await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
    print("Tablas verificadas/creadas en la base de datos.")



async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session

