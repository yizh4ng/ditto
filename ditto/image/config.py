from ditto.shape_generator import polygon_generator

# class Config():
#   suffle = True
#   smooth = 0
#   uncentral_offset = (50, 50)
#   radius = 30
#   Square = square.Square_generator(min_size=(0.1,0.1), max_size=(0.3, 0.3),
#                                    num_range=(1,5), height_range=(0.2,10),
#                                    draw_over=True, random_rotate=True)
#
#   Ellipse = ellipse.Ellipse_generator(radius_range=(0.1,0.2), long_axis_range=(1.2, 1.5),
#                                       short_axis_range=(0.5,0.6),
#                                       num_range=(1,5), height_range=(0.2,10),
#                                       draw_over=True, random_rotate=True)

  # Polygon = polygon.Polygon_generator(min_size=(0.2, 0.2), max_size=(0.3, 0.3),
  #                                   num_range=(1, 5), num_point_range=(4, 6),
  #                                   height_range=(0.2, 1), draw_over=True,
  #                                   random_rotate=True)
  # pass


Config = {'img_size':(200,200),
          'radius_range': (30, 31),
          'uncentral_range' : ((60, 61), (60, 61)),
          'suffle': True,
          'smooth':0,
          'draw_over': True,
          # 'tilt_abberation': {'slop_direction':(1,1),
          #                     'slop':0.01,
          #                     'lowest_height': 0},
          # 'Square':{'radius_range': (0.3, 0.3),
          #             'num_range': (1, 2),
          #             'num_point_range':(4, 6),
          #             'height_range': (0, 1),
          #             'random_rotate': True},
          # 'Ellipse':{'radius_range':(0.1, 0.2),
          #            'long_axis_range':(1.2, 1.5),
          #            'short_axis_range': (0.5, 0.6),
          #            'num_range':(1, 2),
          #            'height_range':(0, 1),
          #            'random_rotate':True
          #           },
          'Polygon': {'radius_range': (0.2, 0.2),
                      'num_range': (4, 5),
                      'num_point_range':(3, 4),
                      'height_range': (0, 1),
                      'random_rotate': True,
                      'irregular': False},
          }
