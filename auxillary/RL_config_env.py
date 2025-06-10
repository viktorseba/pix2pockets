# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 09:07:19 2023

@author: jonas
"""
import numpy as np
import matplotlib.pyplot as plt
# ======== UNIVERSAL ========#

VIEW_MIRRORS = False

if VIEW_MIRRORS: 
    WINDOW_MULTIPLIER = 2.65  # 2.65
    ZOOM = 0.6  #0.6
else: 
    WINDOW_MULTIPLIER = 1  # 3
    ZOOM = 2
    

# FPS = 60  # 500
ANGLE_PRECISION = 100  # 10

THETA_LIMIT = 80 #angle of collision hitpoints

N_ANGLES = 360 * ANGLE_PRECISION  #
# FORCE = int(300 * ZOOM) # 700  
# FORCE = int(30 * ZOOM)
FORCE = 30  # 30

SCREEN_HEIGHT = int(410 * ZOOM)
SCREEN_WIDTH = int(735 * ZOOM)


FULL_SCREEN_WIDTH = SCREEN_WIDTH * WINDOW_MULTIPLIER
FULL_SCREEN_HEIGHT = SCREEN_HEIGHT * WINDOW_MULTIPLIER


BORDER_WIDTH = int(29.5 * ZOOM)

TABLE_H = SCREEN_HEIGHT - (2 * BORDER_WIDTH)
TABLE_W = SCREEN_WIDTH - (2 * BORDER_WIDTH)

CENTER_H = (SCREEN_HEIGHT * WINDOW_MULTIPLIER) / 2
CENTER_W = (SCREEN_WIDTH * WINDOW_MULTIPLIER) / 2

TABLE_SPACE = ((CENTER_W - TABLE_W / 2, CENTER_H - TABLE_H / 2),
               (CENTER_W + TABLE_W / 2, CENTER_H - TABLE_H / 2),
               (CENTER_W + TABLE_W / 2, CENTER_H + TABLE_H / 2),
               (CENTER_W - TABLE_W / 2, CENTER_H + TABLE_H / 2))

NUM_SHOTS = 100
# ======== BALLS ======== #
BALL_MASS = 10
BALL_FRICTION = 0.9  # 0.9
BALL_ELASTICITY = 0.95  # 0.95
BALL_DAMPING_THRESHOLD = 0.90  # 0.6
BALL_RADIUS = int(7 * ZOOM)  # 7

# ======== POCKETS ======== #

POCKET_RADIUS = int(15 * ZOOM)  # 15
POCKET_CORNER_OFFSET = int(5.88 * ZOOM)

POCKET_CENTERS = ((CENTER_W - TABLE_W / 2 + POCKET_CORNER_OFFSET, CENTER_H - TABLE_H / 2 + POCKET_CORNER_OFFSET),
                  (CENTER_W + TABLE_W / 2 - POCKET_CORNER_OFFSET, CENTER_H - TABLE_H / 2 + POCKET_CORNER_OFFSET),
                  (CENTER_W + TABLE_W / 2 - POCKET_CORNER_OFFSET, CENTER_H + TABLE_H / 2 - POCKET_CORNER_OFFSET),
                  (CENTER_W - TABLE_W / 2 + POCKET_CORNER_OFFSET, CENTER_H + TABLE_H / 2 - POCKET_CORNER_OFFSET),
                  (CENTER_W, CENTER_H + TABLE_H / 2),
                  (CENTER_W, CENTER_H - TABLE_H / 2))

# POCKET_TARGETS = []

# ======== RAILS ======== #
CUSHION_FRICTION = 0.5  # 0.5
CUSHION_ELASTICITY = 0.8

CUSHION_WIDTH = int(11 * ZOOM)
CUSHION_OFFSET_MIDDLE = int(3.675 * ZOOM)
CUSHION_OFFSET_CORNER = int(33 * ZOOM)
HOLE_RADIUS = int(19 * ZOOM)
HOLE_RADIUS_MIDDLE = int(16.17 * ZOOM)


CUSHION_INNER_LINES = [CENTER_W - TABLE_W/2 + CUSHION_WIDTH + BALL_RADIUS, # left side
                        CENTER_W + TABLE_W/2 - CUSHION_WIDTH - BALL_RADIUS, # right side
                        CENTER_H + TABLE_H/2 - CUSHION_WIDTH - BALL_RADIUS, # bottom
                        CENTER_H - TABLE_H/2 + CUSHION_WIDTH + BALL_RADIUS] # top

# left-bot,  right-bot,  right-top,  left-top,  mid-top,  mid-bot
TARGET_POSITIONS = [(CUSHION_INNER_LINES[0], CUSHION_INNER_LINES[2]),   # left-bot
                    (CUSHION_INNER_LINES[1], CUSHION_INNER_LINES[2]),   # right-bot
                    (CUSHION_INNER_LINES[1], CUSHION_INNER_LINES[3]),   # right-top
                    (CUSHION_INNER_LINES[0], CUSHION_INNER_LINES[3]),   # left-top
                    (CENTER_W, CUSHION_INNER_LINES[3]),                 # mid-top
                    (CENTER_W, CUSHION_INNER_LINES[2])]                 # mid-bot

CUSHION_INNER_LINES_FOR = [CUSHION_INNER_LINES[1]]

                    # left side
CUSHION_POSITIONS=[((CENTER_W - TABLE_W/2,                      CENTER_H - TABLE_H/2 + HOLE_RADIUS),
            
                    (CENTER_W - TABLE_W/2 + CUSHION_WIDTH,      CENTER_H - TABLE_H/2 + CUSHION_OFFSET_CORNER),
                    
                    (CENTER_W - TABLE_W/2 + CUSHION_WIDTH,      CENTER_H + TABLE_H/2 - CUSHION_OFFSET_CORNER),
                    
                    (CENTER_W - TABLE_W/2,                      CENTER_H + TABLE_H/2 - HOLE_RADIUS) ),
                   
                   # right side
                   ((CENTER_W + TABLE_W/2,                      CENTER_H + TABLE_H/2 - HOLE_RADIUS),
                                       
                    (CENTER_W + TABLE_W/2 - CUSHION_WIDTH,      CENTER_H + TABLE_H/2 - CUSHION_OFFSET_CORNER),
                                       
                    (CENTER_W + TABLE_W/2 - CUSHION_WIDTH,      CENTER_H - TABLE_H/2 + CUSHION_OFFSET_CORNER),
                                       
                    (CENTER_W + TABLE_W/2,                      CENTER_H - TABLE_H/2 + HOLE_RADIUS) ), 
                   
                   # bottom right
                   ((CENTER_W + TABLE_W/2 - HOLE_RADIUS,                    CENTER_H + TABLE_H/2),
                                       
                    (CENTER_W + TABLE_W/2 - CUSHION_OFFSET_CORNER,          CENTER_H + TABLE_H/2 - CUSHION_WIDTH),
                                       
                    (CENTER_W + HOLE_RADIUS_MIDDLE + CUSHION_OFFSET_MIDDLE, CENTER_H + TABLE_H/2 - CUSHION_WIDTH),
                                       
                    (CENTER_W + HOLE_RADIUS_MIDDLE,                         CENTER_H + TABLE_H/2) ), 
                   
                   # bottom left
                   ((CENTER_W - HOLE_RADIUS_MIDDLE,                         CENTER_H + TABLE_H/2),
                                       
                    (CENTER_W - HOLE_RADIUS_MIDDLE - CUSHION_OFFSET_MIDDLE, CENTER_H + TABLE_H/2 - CUSHION_WIDTH),
                                       
                    (CENTER_W - TABLE_W/2 + CUSHION_OFFSET_CORNER,          CENTER_H + TABLE_H/2 - CUSHION_WIDTH),
                                       
                    (CENTER_W - TABLE_W/2 + HOLE_RADIUS,                    CENTER_H + TABLE_H/2) ), 
                   
                   # top right
                   ((CENTER_W + TABLE_W/2 - HOLE_RADIUS,                    CENTER_H - TABLE_H/2),
                                       
                    (CENTER_W + TABLE_W/2 - CUSHION_OFFSET_CORNER,          CENTER_H - TABLE_H/2 + CUSHION_WIDTH),
                                       
                    (CENTER_W + HOLE_RADIUS_MIDDLE + CUSHION_OFFSET_MIDDLE, CENTER_H - TABLE_H/2 + CUSHION_WIDTH),
                                       
                    (CENTER_W + HOLE_RADIUS_MIDDLE,                         CENTER_H - TABLE_H/2) ), 
                   
                   # top left
                   ((CENTER_W - HOLE_RADIUS_MIDDLE,                         CENTER_H - TABLE_H/2),
                                       
                    (CENTER_W - HOLE_RADIUS_MIDDLE - CUSHION_OFFSET_MIDDLE, CENTER_H - TABLE_H/2 + CUSHION_WIDTH),
                                       
                    (CENTER_W - TABLE_W/2 + CUSHION_OFFSET_CORNER,          CENTER_H - TABLE_H/2 + CUSHION_WIDTH),
                                       
                    (CENTER_W - TABLE_W/2 + HOLE_RADIUS,                    CENTER_H - TABLE_H/2) ), 
                   
                   
                   ]

LOWER_X = CENTER_W - TABLE_W / 2 + CUSHION_WIDTH + BALL_RADIUS
LOWER_Y = CENTER_H - TABLE_H / 2 + CUSHION_WIDTH + BALL_RADIUS

UPPER_X = CENTER_W + TABLE_W / 2 - CUSHION_WIDTH - BALL_RADIUS
UPPER_Y = CENTER_H + TABLE_H / 2 - CUSHION_WIDTH - BALL_RADIUS

DIAGONAL = np.sqrt(((UPPER_X - LOWER_X)**2 + (UPPER_Y - LOWER_Y)**2))

b = np.array([c[1:3] for c in CUSHION_POSITIONS]).reshape(-1, 2)
sets = [(1, 7), (4, 2), (8, 3), (0, 11), (10, 9), (6, 5)]
# TARGET_POSITIONS = np.array([np.array([b[s[0]], b[s[1]]]).mean(axis=0) for s in sets])
# TARGET_POSITIONS[5][1] += BALL_RADIUS
# TARGET_POSITIONS[4][1] -= BALL_RADIUS
CUSHION_CORNERS = [[b[s[0]], b[s[1]]] for s in sets]

# ======== COLORS ========#
BLACK = (0, 0, 0)
GRAY = (127, 127, 127)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
PURPLE = (255, 153, 255)

TABLE_COLOR = (0, 100, 0)
BG_COLOR = (30, 30, 30)
TABLE_SIDE_COLOR = (0, 70, 0)
SOLID_COLOR = (250, 130, 0)
STRIPE_COLOR = (0, 110, 220)

SUIT_NAMES = ["Solids", "Stripes", "Cue", "Black"]
SUIT_COLORS = ["Orange", "Blue", "White", "Black"]
SUIT_COLOR = [PURPLE,SOLID_COLOR,STRIPE_COLOR,WHITE,BLACK]

DRAW_SHOT_COLOR = [(255, 178, 102), (153, 204, 255), (255, 255, 153), (255, 153, 255)]
#                      orange-ish ,     blue-ish   ,  bright-yellow ,        purple

BESTSHOTCOLOR = YELLOW
SHOTCOLOR = BLACK
INTERUPTEDCOLOR = RED
