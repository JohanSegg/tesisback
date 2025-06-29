# tests/conftest.py

import pytest
import pytest_asyncio
from typing import AsyncGenerator
from httpx import ASGITransport, AsyncClient

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from fastapi.testclient import TestClient

# Importa tu aplicación FastAPI y la función get_session
from main import app
from main import get_session


from models import Trabajador # Importamos un modelo para crear datos de prueba

# Usamos una base de datos en memoria para las pruebas, es más rápido y se aísla.
# NOTA: SQLite en memoria no soporta todas las características de PostgreSQL.
# Para pruebas más complejas, se podría usar una base de datos PostgreSQL de prueba.
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

engine = create_async_engine(TEST_DATABASE_URL, echo=True, future=True)
async_session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# Esta es una dependencia de sesión SOBREESCRITA para usar la base de datos de prueba
async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session

# Sobreescribimos la dependencia en nuestra aplicación
app.dependency_overrides[get_session] = override_get_session


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Fixture que crea y limpia la base de datos para cada función de prueba.
    """
    # Crear todas las tablas antes de la prueba
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    # Proporcionar la sesión a la prueba
    async with async_session_factory() as session:
        yield session
    
    # Limpiar la base de datos después de la prueba
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Fixture que proporciona un cliente HTTP asíncrono para hacer peticiones a la API.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac