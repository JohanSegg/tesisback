# conftest.py
# Entorno de pruebas para la BD.

import pytest
import pytest_asyncio
from typing import AsyncGenerator
from httpx import ASGITransport, AsyncClient

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from fastapi.testclient import TestClient

# Importa la aplicacion fastapi (app) y la funcion get_session
from main import app
from main import get_session
from models import Trabajador # Importa modelo "trabajador" para crear datos de prueba

# Base de datos temporal en memoria para las pruebas.
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

engine = create_async_engine(TEST_DATABASE_URL, echo=True, future=True)
async_session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# Esta es una dependencia de sesión que se sobreescribe para usar la base de datos de prueba
async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session

app.dependency_overrides[get_session] = override_get_session # reemplaza la dependencia en app de main.py

@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Fixture (funcion) que crea y limpia la BD para cada función de prueba.
    """
    # Crear todas las tablas de la BD antes de la prueba
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    # delega la sesion 
    async with async_session_factory() as session:
        yield session
    
    # Limpia la BD luego de la prueba
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Fixture que proporciona un cliente HTTP asíncrono para hacer peticiones a la API.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
