# MLOps - Proyecto de Operacionalización de Machine Learning

**Proyecto final presentado en la materia MLOps - Operacionalización de Machine Learning en el Master Deep Learning en la UPM**

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Interfaz NLP](#interfaz-nlp)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Despliegue](#despliegue)
- [Contribución](#contribución)
- [Licencia](#licencia)

## 📖 Descripción

Este proyecto implementa un pipeline completo de MLOps para el procesamiento y análisis de texto utilizando técnicas de Natural Language Processing (NLP). El objetivo es demostrar las mejores prácticas en la operacionalización de modelos de Machine Learning, incluyendo:

- Desarrollo y entrenamiento de modelos NLP
- Containerización con Docker
- CI/CD con GitHub Actions
- Monitoreo y logging
- API REST para inferencia
- Interfaz de usuario interactiva

## ✨ Características

- 🤖 **Modelos NLP**: Implementación de modelos para análisis de sentimientos, clasificación de texto y más
- 🔄 **Pipeline automatizado**: Flujo completo desde entrenamiento hasta despliegue
- 🐳 **Containerización**: Aplicación dockerizada para fácil despliegue
- 🚀 **CI/CD**: Integración y despliegue continuo con GitHub Actions
- 📊 **Monitoreo**: Logging y métricas de rendimiento
- 🌐 **API REST**: Endpoints para inferencia de modelos
- 💻 **Interfaz gráfica**: Interfaz de usuario amigable para interactuar con los modelos

## 🛠️ Requisitos

### Requisitos del Sistema
- Python 3.9+
- Docker
- Git

### Dependencias principales
- FastAPI
- scikit-learn
- pandas
- numpy
- streamlit (para la interfaz)
- uvicorn

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/iamam07/MLOps.git
cd MLOps
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con tus configuraciones específicas
```

## 💡 Uso

### Ejecución local
```bash
# Iniciar el servidor API
python main.py

# Iniciar la interfaz NLP (en otra terminal)
python interfaceNLP.py
```

### Con Docker
```bash
# Construir la imagen
docker build -t mlops-app .

# Ejecutar el contenedor
docker run -p 8000:8000 -p 8501:8501 mlops-app
```

## 🎯 Interfaz para Simitud

La interfaz proporciona una interfaz gráfica interactiva para trabajar con un modelos de procesamiento de lenguaje natural exactamente en temas de Similitud Semántica entre dos oraciones.

Nota: A pesar de que la interface esta preparada para otras funciones, en esta version solo esta disponible la opcion de Similitud Semántica la cual se puede seleccionar en la opcion lateral donde tambien existen otras opciones disponible.

### Características de la Interfaz

- **Similitud Semántica**: Determina el nivel de Similitud Semántica entre dos oraciones.

### Guía de Uso de la Interfaz

#### 1. Iniciar la Interfaz
```bash
python interfaceNLP.py
```
La interfaz se abrirá automáticamente en tu navegador en `http://localhost:8501`

#### 2. Análisis de Texto Individual

1. **Selecciona el tipo de opcion** en el menú desplegable lateral:
   - Similitud Semántica

#### 3. Procesamiento de las oraciones 

1. **Indicar las Oraciones** completa los campos correspondiente a las dos oraciones a la que se le estara realizando la Similitud Semántica.

2. **Ejecuta la Comparacion** y se desplegara en la parte de arriba como si fuera un chat el resultado de la comparacion semantica de ambas oraciones.


#### 4. Configuración Avanzada

- **Ajustar umbral de confianza**: Modifica el nivel mínimo de confianza para las predicciones
- **Seleccionar idioma**: Cambia el idioma de procesamiento
- **Personalizar salida**: Elige el formato de los resultados exportados


## 📁 Estructura del Proyecto

```
MLOps/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   └── models.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   └── ml/
│       ├── __init__.py
│       ├── models.py
│       ├── preprocessing.py
│       └── training.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_models.py
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── main.py
├── interfaceNLP.py
└── README.md
```

## 🚀 Despliegue

### GitHub Actions CI/CD

El proyecto incluye un pipeline de CI/CD que:

1. **Ejecuta tests** automáticamente en cada push
2. **Construye la imagen Docker** 
3. **Despliega a GitHub Container Registry**
4. **Actualiza el entorno de producción**

### Despliegue Manual

```bash
# Construir y tagear la imagen
docker build -t ghcr.io/iamam07/mlops-app:latest .

# Subir al registro
docker push ghcr.io/iamam07/mlops-app:latest

# Desplegar en producción
docker run -d -p 8000:8000 -p 8501:8501 ghcr.io/iamam07/mlops-app:latest
```

## 🔧 Configuración

### Variables de Entorno

```bash
# .env
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=./data/models/
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### Configuración de Modelos

Los modelos se configuran en `app/core/config.py`:

```python
MODELS_CONFIG = {
    "sentiment_analysis": {
        "model_path": "models/sentiment_model.pkl",
        "threshold": 0.7
    }
}
```

## 🤝 Contribución

1. **Fork** el proyecto
2. **Crea** una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre** un Pull Request

### Estándares de Código

- Usar **Black** para formateo
- Seguir **PEP 8**
- Incluir **docstrings** en todas las funciones
- Escribir **tests** para nuevas funcionalidades

## 📊 Métricas y Monitoreo

El proyecto incluye:

- **Logging estructurado** con diferentes niveles
- **Métricas de rendimiento** de los modelos
- **Monitoreo de salud** de la API
- **Alertas** por email en caso de fallos

## 🔗 Enlaces Útiles

- [Documentación de FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [MLOps Best Practices](https://ml-ops.org/)

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👨‍💻 Autor

**Miguel A. Martinez** - *Proyecto Final MLOps*
- GitHub: [@iamam07](https://github.com/iamam07)
- LinkedIn: [Tu perfil de LinkedIn]



⭐ **¡Si este proyecto te resulta útil, considera darle una estrella!** ⭐