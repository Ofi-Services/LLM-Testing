import logging

def setup_logger(LOGGING_FILE: str = "log.log"):
    # Configuración de logging
    LOGGING_LEVEL = logging.INFO # Cambiar a logging.DEBUG para más detalles
    LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s: %(message)s'
    """Configura el logging básico para la aplicación."""
    logging.basicConfig(filename=LOGGING_FILE, level=LOGGING_LEVEL, format=LOGGING_FORMAT)
    return 