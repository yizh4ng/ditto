from ditto.shape_generator import polygon_generator



Config = {'img_size':(500, 500),
          'radius_range': (50, 51),
          'uncentral_range' : ((60, 61), (60, 61)),
          'suffle': True,
          'smooth':2,
          'draw_over': False,
          # 'tilt_abberation': {'slop_direction':(0,1),
          #                     'slop':0.01,
          #                     'lowest_height': 0},
          # 'Square':{'radius_range': (0.3, 0.3),
          #             'num_range': (1, 2),
          #             'num_point_range':(4, 6),
          #             'height_range': (0, 1),
          #             'random_rotate': True},
          'Ellipse':{'radius_range':(0.1, 0.2),
                     'long_axis_range':(1.0, 1.1),
                     'short_axis_range': (0.9, 1.0),
                     'num_range':(4, 5),
                     'height_range':(5, 10),
                     'random_rotate':True
                    },
          # 'Polygon': {'radius_range': (0.2, 0.2),
          #             'num_range': (4, 5),
          #             'num_point_range':(3, 4),
          #             'height_range': (5, 10),
          #             'random_rotate': True,
          #             'irregular': False},
          }