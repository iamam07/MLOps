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

## 🎯 Interfaz NLP

La interfaz NLP (`interfaceNLP.py`) proporciona una interfaz gráfica interactiva para trabajar con los modelos de procesamiento de lenguaje natural.

### Características de la Interfaz

- **Análisis de Sentimientos**: Determina si un texto es positivo, negativo o neutral
- **Clasificación de Texto**: Clasifica textos en categorías predefinidas
- **Procesamiento en Lote**: Carga y procesa múltiples textos desde archivos
- **Visualización de Resultados**: Gráficos y métricas de los análisis realizados

### Guía de Uso de la Interfaz

#### 1. Iniciar la Interfaz
```bash
python interfaceNLP.py
```
La interfaz se abrirá automáticamente en tu navegador en `http://localhost:8501`

#### 2. Análisis de Texto Individual

1. **Selecciona el tipo de análisis** en el menú desplegable:
   - Análisis de Sentimientos
   - Clasificación de Texto
   - Extracción de Entidades

2. **Ingresa tu texto** en el área de texto proporcionada

3. **Haz clic en "Analizar"** para obtener los resultados

4. **Visualiza los resultados** que incluyen:
   - Predicción del modelo
   - Confianza de la predicción
   - Gráficos de probabilidades

#### 3. Procesamiento en Lote

1. **Ve a la sección "Procesamiento en Lote"**

2. **Carga tu archivo** (formatos soportados: .txt, .csv, .json):
   ```
   Ejemplo de formato CSV:
   texto,etiqueta_real
   "Me encanta este producto",positivo
   "No me gusta nada",negativo
   ```

3. **Selecciona las columnas** relevantes si es un CSV

4. **Ejecuta el análisis** y descarga los resultados

#### 4. Comparación de Modelos

1. **Accede a la sección "Comparación"**

2. **Selecciona múltiples modelos** para comparar

3. **Ingresa el texto de prueba**

4. **Compara los resultados** lado a lado con métricas de rendimiento

#### 5. Configuración Avanzada

- **Ajustar umbral de confianza**: Modifica el nivel mínimo de confianza para las predicciones
- **Seleccionar idioma**: Cambia el idioma de procesamiento
- **Personalizar salida**: Elige el formato de los resultados exportados

### Ejemplos de Uso

#### Análisis de Sentimientos
```python
# Texto de ejemplo
texto = "¡Excelente servicio! Muy recomendado."

# Resultado esperado
{
    "sentimiento": "positivo",
    "confianza": 0.95,
    "probabilidades": {
        "positivo": 0.95,
        "neutral": 0.04,
        "negativo": 0.01
    }
}
```

#### Clasificación de Texto
```python
# Texto de ejemplo
texto = "¿Cuál es el horario de atención al cliente?"

# Resultado esperado
{
    "categoria": "soporte_cliente",
    "confianza": 0.87,
    "subcategorias": ["horarios", "informacion_general"]
}
```

### Solución de Problemas

#### Error de Conexión
- Verifica que el servidor API esté ejecutándose en `http://localhost:8000`
- Revisa que no haya conflictos de puertos

#### Rendimiento Lento
- Reduce el tamaño del lote de procesamiento
- Verifica los recursos disponibles del sistema

#### Errores de Formato
- Asegúrate de que los archivos tengan la codificación UTF-8
- Verifica que el formato del CSV sea correcto

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
    },
    "text_classification": {
        "model_path": "models/classifier_model.pkl",
        "categories": ["categoria1", "categoria2", "categoria3"]
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

**Andrés Mejía** - *Proyecto Final MLOps*
- GitHub: [@iamam07](https://github.com/iamam07)
- LinkedIn: [Tu perfil de LinkedIn]

## 🙏 Agradecimientos

- Universidad Politécnica de Madrid (UPM)
- Master en Deep Learning
- Profesores y compañeros del programa MLOps

---

⭐ **¡Si este proyecto te resulta útil, considera darle una estrella!** ⭐