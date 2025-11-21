import os

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP'}

def allowed_file(filename: str) -> bool:
    """Verificar si el archivo tiene extensión permitida"""
    if not filename or '.' not in filename:
        return False
    return get_file_extension(filename) in ALLOWED_EXTENSIONS

def get_file_extension(filename: str) -> str:
    """Obtener extensión del archivo"""
    return filename.rsplit('.', 1)[1] if '.' in filename else ''
