# test_main.py

import pytest
from httpx import AsyncClient
from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession
from models import Trabajador

# Marcar todas las pruebas en este archivo para que usen asyncio
pytestmark = pytest.mark.asyncio


# --- Pruebas para Endpoints de Autenticación (US01) ---

async def test_register_user_success(client: AsyncClient):
    """
    Prueba que un usuario puede registrarse exitosamente.
    """
    response = await client.post(
        "/register/",
        json={
            "nombre": "Andres Galvez",
            "username": "agalvez",
            "password": "a_secure_password"
        }
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["username"] == "agalvez"
    assert "password" not in data  # ¡Muy importante! Nunca devolver la contraseña


async def test_register_user_duplicate_username(client: AsyncClient):
    """
    Prueba que el sistema previene el registro con un nombre de usuario duplicado
    y devuelve un código de estado 409 Conflict.
    """
    user_payload = {
        "nombre": "Johan Segura",
        "username": "jsegura",
        "password": "another_password"
    }

    # Primero, creamos un usuario. Esperamos que sea exitoso (201 Created).
    response1 = await client.post("/register/", json=user_payload)
    assert response1.status_code == status.HTTP_201_CREATED

    # Luego, intentamos registrar otro con el mismo username.
    response2 = await client.post("/register/", json=user_payload)
    
    # Ahora, esperamos el código de error 409 Conflict y el mensaje correcto.
    assert response2.status_code == status.HTTP_409_CONFLICT
    assert response2.json()["detail"] == "El nombre de usuario ya existe. Por favor, elige otro."


async def test_login_user_success(client: AsyncClient):
    """
    Prueba que un usuario registrado puede iniciar sesión con credenciales correctas.
    """
    # 1. Registrar un usuario de prueba
    user_payload = {
        "nombre": "Test User",
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


async def test_login_user_wrong_password(client: AsyncClient):
    """
    Prueba que el inicio de sesión falla si la contraseña es incorrecta.
    """
    # 1. Registrar usuario
    user_payload = {
        "nombre": "Wrong Pass User",
        "username": "wrongpass",
        "password": "correct_password"
    }
    await client.post("/register/", json=user_payload)

    # 2. Intentar login con contraseña incorrecta
    login_payload = {
        "username": "wrongpass",
        "password": "incorrect_password"
    }
    response = await client.post("/login/", json=login_payload)

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Credenciales inválidas."


async def test_login_user_not_found(client: AsyncClient):
    """
    Prueba que el inicio de sesión falla si el usuario no existe.
    """
    response = await client.post(
        "/login/",
        json={"username": "nonexistentuser", "password": "any_password"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

# --- Pruebas para Endpoints de Sesiones (US03) ---
# Nota: Para probar endpoints protegidos, necesitarías un sistema de tokens (JWT).
# Como tu endpoint de login no devuelve un token, no podemos probar esto de forma
# unitaria sin modificar el código. La prueba se enfocará en la creación.

async def test_start_session_success(client: AsyncClient):
    """
    Prueba que se puede iniciar una nueva sesión para un trabajador existente.
    """
    # Primero, creamos un trabajador para asociarle la sesión
    reg_response = await client.post(
        "/register/",
        json={"nombre": "Session User", "username": "sessionuser", "password": "pw"}
    )
    trabajador_id = reg_response.json()["trabajador_id"]

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