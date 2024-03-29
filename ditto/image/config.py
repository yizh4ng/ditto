from ditto.shape_generator import polygon_generator
import numpy as np


Config = {#'img_size':(1024, 1280),
          'img_size':(1024, 1280),
          'radius_range': (50, 51),
          'uncentral_range' : ((60, 61), (60, 61)),
          'source_light_pos':(0,0,1000000),
          'ref_light_height':1000000,
          'ref_light_offset': np.sqrt(1280000 ** 2 + 2000000**2),
          # 'ref_light_offset': np.sqrt(12 ** 2 + 20**2),
          'ref_light_angle': (-np.pi/2, 0),
          # 'ref_light_angle': np.arctan(-2000000/1280000),
          # 'first_phase': (-0.5 * np.pi, 0.5 * np.pi),
          'first_phase': (0,0),
          'frequency':(1.160, 1.2),#1.180
          # 'frequency':(100, 100),#1.180
          'suffle': True,
          'smooth':0,
          'draw_over': False,
          'with_abberation': False,
          'positive_image':False,
          'normalize': False,
          'binary':False,
          # 'binary':False,
          # 'tilt_abberation': {'slop_direction':(0,1),
          #                     'slop':0.01,
          #                     'lowest_height': 0},
          # 'quadratic_aberration': {'random_rotate':True,
          #                         'long_axis_range':(1,2),
          #                         'short_axis_range': (0.5,1),
          #                          'height_range': [0.5, 1],
          #                          'radius_range': [5, 10]},
          # 'Square':{'radius_range': (0.3, 0.3),
          #             'num_range': (2, 4),
          #             'num_point_range':(4, 6),
          #             'height_range': [[-5.0, -3.5], [3.5, 5.0]],
          #             'random_rotate': True},
          # 'Ellipse':{'radius_range':(0.01, 0.2),
          #            'long_axis_range':[[1.0, 1.1]],
          #            'short_axis_range': [[0.9, 1.0]],
          #            'num_range':(1, 2),
          #            # 'height_range':[[-5.0, -3.5], [3.5, 5.0]],
          #            'height_range':[[0.5, 1]],
          #            'random_rotate':[[0, 30]],
          #            'center':True,
          #            'uniform_height':False
          #           },
          'Zernike':{'zernike_coefficient':None}
          # 'Polygon': {'radius_range': (0.01, 0.2),
          #             'num_range': (3, 5),
          #             'num_point_range':(3, 5),
          #             'height_range': [[-5.0, -3.5], [3.5, 5]],
          #             'random_rotate': True,
          #             'irregular': True
          #          },
          # 'sinoid_texture': {  'first_phase': (0, 0.5 * np.pi),
          #                      'frequency': (0.5, 0.7),
          #                      'height_offset': (1, 2),
          #                      'amplitude': (0.5, 1),
          #                      'angle':(0, 0.5 * np.pi)
          #          }
          }