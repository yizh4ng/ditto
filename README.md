# ditto
This is a package to generate simple shapes, such as ellipses and polygons, with rich customization.

This package also supports video generation, where the shapes can move and rotate randomly.

This package can encode the images and video in fringes.

![Ditto](https://user-images.githubusercontent.com/50898990/161477872-8ec8b8e6-cd86-4264-8922-63e4a6a7b224.png)


# Update
Support textures embedding on the generated shapes.
![ground-truth-Train Set-0](https://user-images.githubusercontent.com/50898990/161478543-185ca573-c05f-449e-a206-308efe947ec9.png)

# Dependency
numpy, matplotlib, opencv, scipy

lambo is optional for visualiation

# Demo
To generate a single image and its interferogram:
```
from ditto import Painter, ImageConfig

p = Painter(ImageConfig)

p.paint_samples()
```

Ground truths and fringes can be extracted from p.ground_truth and p.fringes.

To generate a video of images and their interferograms:
```
from ditto import ImageConfig, VideoConfig, VideoGenerator

vg = VideoGenerator(ImageConfig, VideoConfig)

vg.generate()
```

Then the grounds truthes and fringes stack can be extracted from vg.img_stack and vg.fringe_stack.

Note that the video generator will automatically use the image config.

# Todo

Zernike polynomial abberation.

Background noise simulation for imaging systems.

