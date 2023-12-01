from dataclasses import dataclass
from typing import Optional

from pathlib import Path


source_dir = Path(__file__).parent.parent


@dataclass
class FlaskConfig:
    
    image_dir_path: str = str(source_dir / 'images')
    save_path: Optional[str] = str(source_dir / 'outputs')
    traverse: bool = True
    n: int = 42

    port: Optional[int] = None
    host: Optional[str] = None
    debug: Optional[bool] = None
    threaded: Optional[bool] = True
