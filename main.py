
from generator import ImageGenerator

image_generator = ImageGenerator('./exercise_data', './Labels.json', 12, [32, 32, 3], rotation=False, mirroring=False,shuffle=False)
image_generator.next()
image_generator.next()
image_generator.show()
