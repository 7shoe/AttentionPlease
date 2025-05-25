import yaml
from pathlib import Path
from typing import Union, Dict, Any
import time

def save_yaml_log(filename: Union[str, Path], data: Dict[Any, Any], 
                  create_dirs: bool = True, backup_on_error: bool = True) -> bool:
    """
    Save data structure to YAML file with error handling.
    
    Args:
        filename: Path to the YAML file (string or Path object)
        data: Dictionary or data structure to save as YAML
        create_dirs: Whether to create parent directories if they don't exist
        backup_on_error: Whether to create a backup if the original file exists and write fails
    
    Returns:
        bool: True if successful, False if failed
    
    Example:
        >>> log_data = {
        ...     'metadata': {'timestamp': 1672531200, 'epochs': 10},
        ...     'training_log': {'losses': [1.2, 1.1, 1.0]}
        ... }
        >>> success = save_yaml_log('experiment_log.yaml', log_data)
        >>> print(f"Save successful: {success}")
    """
    try:
        # Convert to Path object for easier handling
        yaml_path = Path(filename)
        
        # Create parent directories if requested
        if create_dirs:
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists and backup is requested
        if backup_on_error and yaml_path.exists():
            backup_path = yaml_path.with_suffix(f'.backup_{int(time.time())}.yaml')
            yaml_path.rename(backup_path)
        
        # Write YAML file
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, 
                     default_flow_style=False,  # Multi-line format
                     indent=2,                  # 2-space indentation
                     sort_keys=False,           # Preserve order
                     allow_unicode=True)        # Support unicode characters
        
        return True
        
    except Exception as e:
        print(f"Error saving YAML to {filename}: {e}")
        return False