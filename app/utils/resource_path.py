import os
import sys

def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and PyInstaller.
    
    When packaged with PyInstaller, resources are extracted to a temp folder.
    This function handles both development and packaged scenarios.
    
    Args:
        relative_path: Path relative to project root (e.g., 'models/model.keras')
    
    Returns:
        Absolute path to the resource
    """
    try:
        #pyinstaller creates a temp folder and stores path in _meipass
        base_path = sys._MEIPASS
    except Exception:
        #development mode - go up two levels from app/utils/ to get project root
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    return os.path.join(base_path, relative_path)

def get_app_data_path(app_name='TuroArnis'):
    """
    Get path to application data directory for storing user data like databases.
    
    Uses %APPDATA% on Windows, ~/.local/share on Linux, ~/Library/Application Support on macOS.
    
    Args:
        app_name: Name of the application folder
    
    Returns:
        Absolute path to app data directory (created if doesn't exist)
    """
    if sys.platform == 'win32':
        app_data = os.getenv('LOCALAPPDATA') or os.getenv('APPDATA')
        if not app_data:
            app_data = os.path.expanduser('~')
    elif sys.platform == 'darwin':
        app_data = os.path.expanduser('~/Library/Application Support')
    else:
        app_data = os.path.expanduser('~/.local/share')
    
    app_dir = os.path.join(app_data, app_name)
    os.makedirs(app_dir, exist_ok=True)
    return app_dir
