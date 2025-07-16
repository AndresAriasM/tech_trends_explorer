import os
from openai import OpenAI
from dotenv import load_dotenv

# Cargar el .env
load_dotenv()

# Obtener la clave de API
api_key = os.getenv("OPENAI_API_KEY")

# Crear el cliente
client = OpenAI(api_key=api_key)

# Hacer una solicitud
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # o "gpt-4" si tienes acceso
    messages=[
        {"role": "user", "content": "¿Cuál es la capital de Francia?"}
    ]
)

# Mostrar la respuesta
print(response.choices[0].message.content)
