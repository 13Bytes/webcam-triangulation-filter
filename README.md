# live webcam triangulation filter
Like all the beautiful polygon pictures - only live and way less nice

Idea based on: http://www.jhclaura.com/triangulation/


## Settings
It uses face-detection to seperate beween fore- and background.


Default-Settings:
```python
POINT_DENSITY = 11
SENSITIVITY = 10

POINT_DENSITY_FACE = 12
SENSITIVITY_FACE = 4
```

## Installation
``` pip install -r requirements.txt ```
``` python3 main.py ```