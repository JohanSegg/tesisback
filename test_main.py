# test_main.py

import pytest
from httpx import AsyncClient
from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession
from models import Trabajador

# Todas las pruebas en este archivo usaran asyncio
pytestmark = pytest.mark.asyncio


# --- Pruebas para Autenticación (US01) ---
# a. Probar registro de usuario
async def test_register_user_success(client: AsyncClient):

    response = await client.post(
        "/register/",
        json={
            "nombre": "Andres Galvez",
            "username": "agalvez",
            "password": "palabra_secreta"
        }
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["username"] == "agalvez"
    assert "password" not in data  # No se "expone" la contraseña

# b. Probar registro de usuario duplicado
async def test_register_user_duplicate_username(client: AsyncClient):

    user_payload = {
        "nombre": "Johan Segura",
        "username": "jsegura",
        "password": "otra_palabra_secreta"
    }

    # Primero creamos un usuario y esperamos que sea exitoso con http 201.
    response1 = await client.post("/register/", json=user_payload)
    assert response1.status_code == status.HTTP_201_CREATED

    # Luego intentamos registrar otro con el mismo username.
    response2 = await client.post("/register/", json=user_payload) 
    
    # esto debe generar un codigo de error 409 conflict
    assert response2.status_code == status.HTTP_409_CONFLICT 
    assert response2.json()["detail"] == "El nombre de usuario ya existe. Por favor, elige otro."

# c. Login exitoso con credenciales correctas
async def test_login_user_success(client: AsyncClient):

    # 1. Registrar un usuario de prueba
    user_payload = {
        "nombre": "usuario_generico",
        "username": "testuser",
        "password": "password123"
    }
    await client.post("/register/", json=user_payload)

    # 2. Intentar iniciar sesión
    login_payload = {
        "username": "testuser",
        "password": "password123"
    }
    response = await client.post("/login/", json=login_payload)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["message"] == "Inicio de sesión exitoso"
    assert "trabajador_id" in data

# d. Login fallido por contraseña incorrecta
async def test_login_user_wrong_password(client: AsyncClient):

    # 1. Registrar usuario
    user_payload = {
        "nombre": "User",
        "username": "testuser",
        "password": "correct_password"
    }
    await client.post("/register/", json=user_payload)

    # 2. Intentar login con contraseña incorrecta
    login_payload = {
        "username": "testuser",
        "password": "incorrect_password"
    }
    response = await client.post("/login/", json=login_payload)

    assert response.status_code == status.HTTP_401_UNAUTHORIZED # <-- Código 401
    assert response.json()["detail"] == "Credenciales inválidas."

# e. Login fallido por usuario no existente
async def test_login_user_not_found(client: AsyncClient):

    response = await client.post(
        "/login/",
        json={"username": "usuario_generico", "password": "contraseña_generica"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED # <-- Código 401

# --- Pruebas para Endpoints de Sesiones (US03) ---
# Se recomienda usar un sistema de tokens (JWT).

# Probar un login nuevo con un trabajador que ya existe
async def test_start_session_success(client: AsyncClient):
  
    # Primero, creamos un trabajador para asociarle la sesión
    reg_response = await client.post(
        "/register/",
        json={"nombre": "peter", "username": "petermg", "password": "secret"}
    )
    trabajador_id = reg_response.json()["trabajador_id"] # id del trabajador

    # Ahora, iniciamos una sesión para ese trabajador
    response = await client.post(
        "/sessions/start/",
        json={"trabajador_id": trabajador_id}
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["trabajador_id"] == trabajador_id
    assert data["estado_grabacion"] == "Iniciada"
    assert data["sesion_id"] is not None
