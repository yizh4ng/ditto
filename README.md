# ditto
So, this is a package to generate simple shapes, such as ellipses and polygons, with rich customization.

This package also supports video generation, where the shapes can move and rotate randomly.

This package can encode the images and video in fringes.

# dependency
numpy, matplotlib, opencv, scipy

lambo is optional for visualiation

# demo
To generate a single image and its interferogram:
```
from ditto import Painter, ImageConfig

p = Painter(**ImageConfig)

p.paint_samples()
```

Ground truths and fringes can be extracted from p.ground_truth and p.fringes.

To generate a video of images and their interferograms:
```
from ditto import ImageConfig, VideoConfig, VideoGenerator

vg = VideoGenerator(ImageConfig, **VideoConfig)

vg.generate()
```

Then the grounds truthes and fringes stack can be extracted from vg.img_stack and vg.fringe_stack.

Note that the video generator will automatically use the image config.

# todo

Zernike polynomial abberation.

For video geneation, when most of shapes move out of the field of view, generate new shapes.

