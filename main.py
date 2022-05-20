from pattern import Circle, Checker, Spectrum
from generator import ImageGenerator

circle = Circle(512, 20, (50, 50))
circle.draw()
circle.show()

checker = Checker(100,25)
checker.draw()
checker.show()

spectrum = Spectrum(100)
spectrum.draw()
spectrum.show()

image_generator = ImageGenerator('./exercise_data', './Labels.json', 12, [32, 32, 3], rotation=False, mirroring=False,shuffle=False)
image_generator.next()
image_generator.next()
image_generator.show()
