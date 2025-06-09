#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:28:10 2023

@author: viktorsebastianpetersen
"""

import gymnasium as gym
import pygame
import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util as pygame_util
import numpy as np
import cv2
import math
import sys
from gymnasium import spaces
# from auxillary.RL_config_env import *
import matplotlib.pyplot as plt
import time
from copy import copy
from typing import Literal, Union

import auxillary.RL_config_env as cfg


class PoolEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
            self,
            algo: Literal['PPO', 'TD3', 'SAC', 'DDPG', 'A2C', 'PPO_masked']='PPO',
            balls_init: Union[np.array, None]=None,
            suit: Literal[1,2]=2,
            training: bool=True,
            fps: int=60,
            num_balls: int=2,
            special_state: Union[int, None]=None,
            obs_type: Literal['vector', 'image']='vector',
            sigma: float=0.0,
            oracle: bool=False,
            game_type: Literal['normal', 'blue_only']='normal',
            bank_shots: bool=True,
            max_num_shots: int=10
    ):
        """_summary_

        Args:
            algo (Literal['PPO', 'TD3', 'SAC', 'DDPG', 'A2C', 'PPO_masked'], optional): What RL algorithm to run. Defaults to 'PPO'.
            
            balls_init (Union[np.array, None], optional): Position and classes of balls in a state. Can be of shape (num_balls x 3) 
            or (num_balls x 3 x num_states). The position of the balls should be in [x,y,c] format, with x,y in the range [0,1] and 
            c is in [1,2,3,4]. Defaults to None.
            
            suit (Literal[1,2], optional): Specifies what class your suit is. Can only be 1 or two (striped or solid). Defaults to 2.
            
            training (bool, optional): Indicates if the environment should be run in training-mode. Defaults to True.
            
            fps (int, optional): Sets the fps. Defaults to 60.
            
            num_balls (int, optional): Sets the number of balls. If balls_init is given, the shape of balls_init replaces this. Defaults to 2.
            
            special_state (Union[int, None], optional): If balls_init is given and has shape (num_balls, 3, num_states), this argument forces 
            the environemnt to always pick this state. Mostly used for debugging. Defaults to None.
            
            obs_type (Literal['vector', 'image'], optional): Type of observation. Use 'image' if you want to utilize a CNN for feature extraction.
            Defaults to 'vector'.
            
            sigma (float, optional): Noise on the angle. Used for a test. Defaults to 0.0.
            
            oracle (bool, optional): Indicates if the Oracle should be used. This will overwrite the actions of the agent completely. Defaults to False.
            
            game_type (Literal['normal', 'blue_only'], optional): 'blue_only' mode sets all balls to blue (ie your suit). Defaults to 'normal'.
            
            bank_shots (bool, optional): Indicates if bank shots should be considerd by the Oracle. Gives rise to more shot opportunities with an
            increase in computational cost. Defaults to True.
            
            max_num_shots (int, optional): Sets the maximum number of shots in an episode before restarting. If training, this is set to 1. Defaults to 10.
        """
        #
        self.algorithm = algo
        self.balls_init = balls_init
        self.suit = suit
        self.training = training
        self.fps = fps
        self.obs_type = obs_type
        self.sigma = sigma
        self.oracle = oracle
        self.game_type = game_type
        self.max_num_shots = 1 if self.training else max_num_shots
        
        self.enemy_suit = 1 if self.suit == 2 else 2

        self.bank_shots = bank_shots #  add extra target pockets
        self.balls = []
        self.ghost_balls = []
        self.ghost_opponents = []
        self.fakehits = []
        self.scores = []
        self.best_score = 0

        self.width = cfg.FULL_SCREEN_WIDTH
        self.height = cfg.FULL_SCREEN_HEIGHT
        self.num_balls = num_balls if balls_init is None else self.balls_init.shape[0]
        self.total_balls = 16
        self.draw_screen = (self.obs_type == "image") or not training
        self.running = True
        self.paused = False
        self.win_list = ['w']
        self.random_state = 0
        self.Continue = False
        self.special_state = special_state

        # Gymnasium environment
        super().__init__()
        print(f"Using {self.algorithm} algorithm")
        if self.algorithm in ["TD3", "DDPG", "SAC"]:
            self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), shape=(2,))
        else:
            self.action_space = spaces.MultiDiscrete([cfg.N_ANGLES, cfg.FORCE])

        self.actions = [[0,0]] 

        if self.obs_type == "vector":
            box_low = np.array([0.0, 0.0, 0] * self.total_balls)
            box_high = np.array([1.0, 1.0, 1] * self.total_balls)
            box_shape = (3 * self.total_balls, )  # obs: [[x1, y1, c1], [x2, y2, c2], ...]
            self.observation_space = spaces.Box(low=box_low,
                                                high=box_high,
                                                shape=box_shape,
                                                dtype=np.float64)
        elif self.obs_type == "image":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(3, self.height, self.width),
                dtype=np.uint8,
            )
        else:
            print("Wrong observation type!!!")

        render_mode = "human"
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = cfg.BALL_DAMPING_THRESHOLD  # 0.83
        self.space.collision_slop = 0.5  # 0.5
        self.space.idle_speed_threshold = 5  # 5
        self.space.sleep_time_threshold = 1e-8  # 1e-8
        self.space.iterations = 20  # 10

        # speed
        self.dt = 30  # 30, 50
        self.step_size = 0.1  # 0.01  0.1
        # self.dt = 1000  # 24
        # self.step_size = 0.01  # 0.45
        self.frame_rate = 1

        # pygame
        if self.draw_screen:
            pygame.init()
            self.screen = pygame.display.set_mode((int(self.width), int(self.height)))
            self.clock = pygame.time.Clock()
            self.draw_options = pygame_util.DrawOptions(self.screen)

        self.wins = 0
        self.loses = 0

        self.total_steps = 0
        self.win_pr_1000 = 0
        self.steps_pr_1000 = 0
        self.no_hitpoint_counter = 0
        self.cue_pocketed = 0
        self.start_time = time.perf_counter()
        self.replay_state = False
        self.view_lines = False
        self.failed_states = []
        self.hit_success = []
        self.best_score = 0
        self.n_hp = [] # Number of hitpoints

        # Collision handlers
        self.ball_collision_handler = self.space.add_collision_handler(1, 1)
        self.ball_collision_handler.begin = self.ball_contacted
        self.ball_collision_handler.post_solve = self.ball_post_solve
        # self.ball_collision_handler.pre_solve = self.ball_pre_solve
        self.pocket_collision_handler = self.space.add_collision_handler(1, 2)
        self.pocket_collision_handler.begin = self.ball_pocketed
        self.cushion_collision_handler = self.space.add_collision_handler(1, 3)
        self.cushion_collision_handler.begin = self.cushion_contacted
        self.cushion_collision_handler.post_solve = self.cushion_post_solve


        self.trash_lines = []

        self.pocket_ids = [0,1,2,3,4,5]
        if self.bank_shots:
            extra_targets = []
            extra_corners = []
            for i,line in enumerate(cfg.CUSHION_INNER_LINES):
                for k,target in enumerate(cfg.TARGET_POSITIONS):
                    dist = target[int(i>=2)] - line
                    dist1 = cfg.CUSHION_CORNERS[k][0][int(i>=2)] - line
                    dist2 = cfg.CUSHION_CORNERS[k][1][int(i>=2)] - line
                    if dist > cfg.POCKET_RADIUS or dist < -cfg.POCKET_RADIUS:
                        self.pocket_ids.append(100*(i+1) + k)
                        if i < 2:
                            extra_targets.append([line-dist,target[1]])
                            extra_corners.append([[line-dist1,cfg.CUSHION_CORNERS[k][0][1]],[line-dist2,cfg.CUSHION_CORNERS[k][1][1]]])
                        else:
                            extra_targets.append([target[0],line-dist])
                            extra_corners.append([[cfg.CUSHION_CORNERS[k][0][0],line-dist1],[cfg.CUSHION_CORNERS[k][1][0],line-dist2]])


            self.target_points = np.concatenate([cfg.TARGET_POSITIONS,extra_targets])
            self.cushion_corners = np.concatenate([cfg.CUSHION_CORNERS,extra_corners])

        else:
            self.target_points = np.array([list(x) for x in cfg.TARGET_POSITIONS])
            self.cushion_corners = np.array(cfg.CUSHION_CORNERS)

        self.window_vectors = []
        self.pocket_value_counter = np.zeros_like(self.target_points[:,0])


        self.draw_stuff = {
            "hit_points": [],
            "hit_points_best": None,
            "draw_hit_points": False
            }
        print("-" * 30)

        # rewards
        self.win_reward = 200
        self.loss_reward = -100
        self.prev_reward = 0

        self.reward = 0

    @property
    def rewardfunc(self):
        return self.reward

    # reward_func
    @rewardfunc.setter
    def rewardfunc(self, addedReward):

        def angle_reward_func(angle):
            # return 100 / (angle + 0.5)
            return 1000 / (angle + 10) - 50

        current_reward = 0
        add = addedReward[0]
        specAdded = False

        if len(addedReward) > 1:
            specAdded = True
            spec = addedReward[1]
            if isinstance(spec, int):
                spec = float(spec)

        if False:
            pass

    # # Termination
        if add == 'win': current_reward = 100
        elif add == 'lose': current_reward = -100

    # # First hit

        # elif add == 'first_hit_suit': current_reward = 10
        elif add == 'hit_suit': current_reward = 10
    #     elif add == 'first_hit_suit_hitdist': current_reward = -spec + BALL_RADIUS  # = -dist*BALL_RADIUS

    #     elif add == 'first_hit_black': current_reward = -10
    #     elif add == 'first_hit_other': current_reward = -5
    #     elif add == 'first_hit_pocket': current_reward = -30
    #     elif add == 'first_hit_cushion': current_reward = -5

    #     elif add == 'first_hit_none_cushion': current_reward = -spec * 50  # = -normdist*50
        # Penalize this harder!
        elif add == 'first_hit_none': current_reward = -spec * 50  # = -normdist*50

    # # Pocketed
    #     duplicated if add == 'pocketed_suit': current_reward = 50 - 5 * self.tracking["cushion_hit"].count(spec) # spec = ball
        elif add == 'pocketed_suit': current_reward = 50
        elif add == 'pocketed_cue': current_reward = -80
    #     if add == 'pocketed_other': current_reward = -10

    # Angle to best hitpoint
        # elif add == 'angle_hitpoint': current_reward = spec  # spec = angle_diff
        # elif add == 'angle_penalty': current_reward = 3 - max(spec, 3)  # Within 3 degrees, no further penalty
    # min angle from target velocity to a pocket
        elif add == 'target_angle_pocket': current_reward = angle_reward_func(spec)
    # Window cue
        # elif add == 'window_cue': current_reward = angle_reward_func(spec)
        # elif add == 'window_target': current_reward = angle_reward_func(spec)
        else:
            if not self.training:
                print("- reward off")
                # pass
            return  # reward is commented out

        self.reward += current_reward
        if not self.training:
            print(add.ljust(25), "{0:3.2f}".format(current_reward), end=' ')

            # print('\t accu:', self.reward,end=' ')

            if specAdded:
                if isinstance(spec, float): print('with spec: ', "{0:3.2f}".format(spec), end=' ')
                else: print('with spec: ball', spec.number, end=' ')
            print()

    @staticmethod
    def ball_post_solve(arbiter, space, data):

        ball1, ball2 = arbiter.shapes

        # Fetch velocities and positions
        v1 = data["tracking"]["last_ball_contact_vel"][ball1.number]
        v2 = data["tracking"]["last_ball_contact_vel"][ball2.number]
        x1 = ball1.body.position
        x2 = ball2.body.position

        # https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional
        dv1 = ((v1 - v2).dot((x1 - x2)) / (x1 - x2).length**2) * (x1 - x2)
        dv2 = ((v2 - v1).dot((x2 - x1)) / (x2 - x1).length**2) * (x2 - x1)

        new_vel1 = v1 - dv1
        new_vel2 = v2 - dv2

        if data["ball_tracking"]["init_velocity"][ball1.number] == Vec2d.zero():
            data["ball_tracking"]["init_velocity"][ball1.number] = ball1.body.velocity
            data["ball_tracking"]["calc_velocity"][ball1.number] = new_vel1
        ball1.body.velocity = new_vel1

        if data["ball_tracking"]["init_velocity"][ball2.number] == Vec2d.zero():
            data["ball_tracking"]["init_velocity"][ball2.number] = ball2.body.velocity
            data["ball_tracking"]["calc_velocity"][ball2.number] = new_vel2
        ball2.body.velocity = new_vel2
        
        
        data["tracking"]["normal_vectors"].append(arbiter.normal)
        return True


    @staticmethod
    def ball_contacted(arbiter, space, data):
        # count bank/carrom shot collisions
        data["tracking"]["ball_collision"] = True
        ball1, ball2 = arbiter.shapes
        s_ball = None
        c_ball = None

        data["tracking"]["ball_contacts"].append((ball1, ball2))
        # Find cue ball (if any)
        if ball1.number == data["cue_ball"].number:
            c_ball = ball1
            s_ball = ball2
        elif ball2.number == data["cue_ball"].number:
            c_ball = ball2
            s_ball = ball1

        if (data["tracking"]["first_cue_contact"] is None) and (s_ball is not None):
            data["tracking"]["first_cue_contact"] = ["ball", s_ball.ballclass]

        if (data["tracking"]["target_ball"] is None) and (s_ball is not None):
            data["tracking"]["target_ball"] = s_ball

        if data["tracking"]["first_suit_collision"] is None:
            if (c_ball is not None) and (s_ball.ballclass == data["tracking"]["suit"]):
                data["tracking"]["first_suit_collision"] = c_ball.body.position

        for contact_point in arbiter.contact_point_set.points:
            data["tracking"]["contact_point"].append((contact_point.point_a, contact_point.point_b))
            data["tracking"]["contact_point_dist"].append(contact_point.distance)

        data["tracking"]["last_ball_contact_vel"][ball1.number] = ball1.body.velocity
        data["tracking"]["last_ball_contact_vel"][ball2.number] = ball2.body.velocity
        return True

    @staticmethod
    def ball_pocketed(arbiter, space, data):
        # arbiter: [ball, pocket]
        ball, pocket = arbiter.shapes

        if ball.ballclass == 3:
            data["tracking"]["cue_ball_pocketed"] = True

            if data["tracking"]["first_cue_contact"] is None:
                data["tracking"]["first_cue_contact"] = ["pocket", 1]

        elif ball.ballclass == 4:
            data["tracking"]["black_ball_pocketed"] = True

        data["tracking"]["ball_pocketed"].append(ball)

        try:
            data["balls"].remove(ball)
        except ValueError:
            pass
        space.remove(ball, ball.body)

        # Change black to your suit when all your balls are pocketed
        if sum([ball.ballclass == data["suit"] for ball in data["balls"]]) == 0:
            for ball in data["balls"]:
                if ball.ballclass == 4:
                    ball.ballclass = data["suit"]

        return False

    @staticmethod
    def cushion_contacted(arbiter, space, data):
        ball, rail = arbiter.shapes
        # count bank/carrom shot collisions
        if ball.ballclass == 3:
            if data["tracking"]["first_cue_contact"] is None:
                data["tracking"]["first_cue_contact"] = ["cushion", 1]

        data["tracking"]["cushion_hit"].append(ball)

        for contact_point in arbiter.contact_point_set.points:
            data["tracking"]["contact_point"].append((contact_point.point_a, contact_point.point_b))
            data["tracking"]["contact_point_dist"].append(contact_point.distance)

        d = ball.body.velocity
        data["tracking"]["last_cushion_vel"] = ball.body.velocity

        return True

    @staticmethod
    def cushion_post_solve(arbiter, space, data):
        ball, rail = arbiter.shapes
        speed = ball.body.velocity.length
        d = data["tracking"]["last_cushion_vel"]
        n = arbiter.normal
        new_vel = (d - 2 * (d.dot(n)) * n).normalized() * speed
        ball.body.velocity = new_vel

    def add_balls(self):
        self.balls = []
        positions = []
        amount_solids = int(np.floor((self.num_balls - 2) / 2))
        amount_stripes = int(np.ceil((self.num_balls - 2) / 2))
        
        self.numblues = amount_stripes
        b_number = 0

        # 1: Stripe, 2: Solid, 3: Cue, 4: Black
        if self.game_type == 'normal' and self.num_balls > 2:
            ball_classes = ([1] * amount_solids) + [4] * (self.num_balls > 1) + ([2] * amount_stripes) + [3]
        else:
            ball_classes = ([2] * (self.num_balls - 1)) + [3]

        if not self.replay_state:
            self.Continue = False

            if self.special_state is None:
                if self.balls_init is None:
                    pass

                elif len(self.balls_init.shape) == 3:
                    # self.random_state = np.random.randint(0, self.balls_init.shape[2])
                    self.random_state = self.total_steps % self.balls_init.shape[2]
            else:
                self.random_state = self.special_state #389

        for i in range(self.num_balls):
            intertia = pymunk.moment_for_circle(cfg.BALL_MASS, 0, cfg.BALL_RADIUS, offset=(0, 0))
            ball_body = pymunk.Body(cfg.BALL_MASS, intertia)

            # initialize positions
            if self.balls_init is not None:
                if len(self.balls_init.shape) == 2:
                    x, y = self.balls_init[i, :2]
                    ball_body.position = [int((cfg.UPPER_X - cfg.LOWER_X - 2)*x + cfg.LOWER_X) + 1,
                                          int((cfg.UPPER_Y - cfg.LOWER_Y - 2)*y + cfg.LOWER_Y) + 1
                                          ]
                    ballclass = int(self.balls_init[i, 2])

                # Below is used for multiple states
                elif len(self.balls_init.shape) == 3:
                    x, y = self.balls_init[i, :2, self.random_state]
                    ball_body.position = [int((cfg.UPPER_X - cfg.LOWER_X - 2)*x + cfg.LOWER_X) + 1,
                                          int((cfg.UPPER_Y - cfg.LOWER_Y - 2)*y + cfg.LOWER_Y) + 1
                                          ]
                    ballclass = int(self.balls_init[i, 2, self.random_state])
                
                else:
                    print(f"Unexpected shape of balls_init! Got shape: {self.balls_init.shape}")

            else:
                # Random position
                new_ball_x = np.random.randint(cfg.LOWER_X, cfg.UPPER_X)
                new_ball_y = np.random.randint(cfg.LOWER_Y, cfg.UPPER_Y)
                overlap = True

                while positions and overlap:  # While other balls and overlap
                    overlap = False

                    for ball in positions:
                        # Calculate distance to every other ball
                        dist = np.sqrt((abs(new_ball_x - ball[0]))**2 + (abs(new_ball_y - ball[1]))**2)

                        if dist <= 2* cfg.BALL_RADIUS:
                            # If overlap, try again
                            new_ball_x = np.random.randint(cfg.LOWER_X, cfg.UPPER_X)
                            new_ball_y = np.random.randint(cfg.LOWER_Y, cfg.UPPER_Y)
                            overlap = True
                            break
                # if no overlap, new ball position is valid
                ball_body.position = [new_ball_x, new_ball_y]
                ballclass = ball_classes[i]

            ball = pymunk.Circle(ball_body, cfg.BALL_RADIUS, offset=(0, 0))
            ball.elasticity = cfg.BALL_ELASTICITY
            ball.friction = cfg.BALL_FRICTION
            ball.collision_type = 1
            ball.ballclass = ballclass

            ball.color = pygame.Color(cfg.SUIT_COLORS[ballclass-1])
            if ballclass == 3:
                ball.number = self.num_balls - 1  # Cue ball be the last ball
                self.cue_ball = ball
                cue_ball_body = ball_body
                positions.append(ball_body.position)
            else:
                ball.number = b_number
                b_number += 1
                positions.append(ball_body.position)
                self.balls.append(ball)
                self.space.add(ball, ball_body)

        self.balls.append(self.cue_ball)
        self.space.add(self.cue_ball, cue_ball_body)

    def add_table(self):
        static_body = self.space.static_body

        self.cushions = []
        for cushion_pos in cfg.CUSHION_POSITIONS:
            cushion = pymunk.Poly(static_body, cushion_pos)
            cushion.color = pygame.Color(cfg.TABLE_SIDE_COLOR)
            cushion.collision_type = 3
            cushion.elasticity = cfg.CUSHION_ELASTICITY
            cushion.friction = cfg.CUSHION_FRICTION
            self.cushions.append(cushion)


        self.pockets = []
        for pocket_loc in cfg.POCKET_CENTERS:
            pocket = pymunk.Circle(static_body, cfg.POCKET_RADIUS, pocket_loc)
            pocket.color = pygame.Color(cfg.BLACK)
            pocket.collision_type = 2
            pocket.elasticity = 0
            pocket.friction = 0
            pocket.position = Vec2d(*pocket_loc)
            self.pockets.append(pocket)

        self.space.add(*self.cushions, *self.pockets)

    def _get_obs(self):
        if self.obs_type == "image":
            img = pygame.surfarray.pixels3d(self.screen)
            img = cv2.resize(img, (self.height, self.width))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = cv2.flip(img, 0)

            # Plot image before changing to channel first
            # plt.imshow(img)
            # plt.show()

            # print(img.shape)
            img = np.moveaxis(img, 2, 0)
            # img = img.reshape(3, self.height, self.width)
            return img

        elif self.obs_type == "vector":
            obs = np.array([[(ball.body.position[0] - cfg.LOWER_X) / (cfg.UPPER_X - cfg.LOWER_X),
                             (ball.body.position[1] - cfg.LOWER_Y) / (cfg.UPPER_Y - cfg.LOWER_Y),
                             ball.ballclass / 4] for ball in self.balls])

            balls_to_fill = self.total_balls - len(self.balls)
            if len(self.balls) == 0:
                # if no balls on table
                obs = (np.repeat(np.array([0, 0, 0]), balls_to_fill, axis=0).reshape(balls_to_fill, 3).flatten())
            elif balls_to_fill > 0:
                # if some balls are pocketed
                obs = np.vstack((obs,
                                 np.repeat(np.array([0, 0, 0]), balls_to_fill, axis=0).reshape(balls_to_fill, 3),)
                                ).flatten()
            else:
                obs = obs.flatten()

            return obs

    def _get_info(self):
        return {**self.tracking, **self.ball_tracking}

    def _get_reward(self):

        if not self.training:
            print('\n\n')
            print('-'*50)
            print("Action".ljust(25),"Angle: {:7.2f}".format(self.angle))
            print("".ljust(25),"Force: {:7.2f}".format(self.force),end="\n\n")

    # Win reward
        # Check if you have any balls left
        if self.win:
            # You have won
            self.rewardfunc = ['win']
            self.wins += 1
            self.win_pr_1000 += 1

    # Loss reward
        if self.lost:
            self.rewardfunc = ['lose']

    # First cue contact reward
        if self.tracking["first_cue_contact"] is None:
            pass
            # Penalty = min dist from cue to balls of your suit
            # if sum([ball.ballclass == self.suit for ball in self.balls]) != 0:
                # dist = np.min([self.cue_ball.body.position.get_distance(ball.body.position) for ball in self.balls if ball.ballclass == self.suit])
                # normdist = dist / DIAGONAL
                # self.rewardfunc = ['first_hit_none', normdist]

        else:
            key, item = self.tracking["first_cue_contact"]
            if key == "ball":
                if item == 4:  # You hit black ball first
                    self.rewardfunc = ['first_hit_black']
                elif item == self.suit:  # You hit your suit first
                    self.rewardfunc = ['first_hit_suit']

                    # Penalty for how far away from a hit point
                    # a, b = self.tracking["contact_point"][0:2]
                    # cp = (a + b) / 2
                    # if len(self.good_hit_points) != 0:
                    #     dist = np.min([cp.get_distance(hit) for hit in self.good_hit_points])
                    #     # print(f"Reward dist (contact point - hit_point): {dist}")
                    #     self.rewardfunc = ['first_hit_suit_hitdist', dist - BALL_RADIUS]

                else:  # You hit other suit first
                    self.rewardfunc = ['first_hit_other']

            elif key == "pocket":
                self.rewardfunc = ['first_hit_pocket']
            elif key == "cushion":
                self.rewardfunc = ['first_hit_cushion']
                # Penalty = min dist from cue to balls of your suit
                if sum([ball.ballclass == self.suit for ball in self.balls]) != 0:
                    dist = np.min([self.cue_ball.body.position.get_distance(ball.body.position) for ball in self.balls if ball.ballclass == self.suit])
                    normdist = dist / cfg.DIAGONAL
                    self.rewardfunc = ['first_hit_none_cushion', normdist]
            else:
                raise Exception("Wrong first_cue_contact key. Something went wrong.")

    # Pocketed balls
        pocketed_balls = self.tracking["ball_pocketed"]

        for ball in pocketed_balls:
            # Pocket rewards
            if ball.ballclass == self.suit:
                # You have pocketed one of your balls
                self.rewardfunc = ['pocketed_suit',ball]

                # penalty for pocketed ball to hit cushion
                # current_reward += -5 * self.tracking["cushion_hit"].count(ball)
            elif ball.ballclass == 3:
                # Cue ball was pocketed
                self.rewardfunc = ['pocketed_cue']

            elif ball.ballclass == 4:
                # Have been accounted for
                pass
            else:
                # Opponents ball was pocketed
                self.rewardfunc = ['pocketed_other']
        # hit suit reward

        if sum([ball.ballclass == self.suit for ball in self.balls]) != 0:  # Check if there are any balls with your suit left
            suit_hit = False
            for ball in self.balls:
                if ball.ballclass == self.suit:
                    if self.ball_tracking["init_velocity"][ball.number] != Vec2d.zero():
                        self.rewardfunc = ['hit_suit']
                        suit_hit = True

            if not suit_hit:
                dist = np.min([self.cue_ball.body.position.get_distance(ball.body.position) for ball in self.balls if ball.ballclass == self.suit])
                normdist = dist / cfg.DIAGONAL
                self.rewardfunc = ['first_hit_none', normdist]


    # Angle reward
        if len(self.good_hit_points) != 0:
            # Angle between shot and closest hit point
            cb = self.ball_tracking["start_positions"][self.cue_ball.number]
            # angle = min([abs(self.cue_shot_vec.get_angle_degrees_between(
            #     hit_point - cb)) for hit_point in self.good_hit_points])
            angle = abs(self.cue_shot_vec.get_angle_degrees_between(self.best_shot))

            self.rewardfunc = ['angle_penalty', angle]
        else:
            # self.no_hitpoint_counter += 1
            self.rewardfunc = ['angle_penalty', 180.0]

    # angle to best hitpoint
        if (len(self.good_hit_points) != 0) and (self.tracking["first_suit_collision"] is not None):

            cb = self.ball_tracking["start_positions"][self.cue_ball.number]

            true_hit = self.tracking["first_suit_collision"] - cb
            dist2hit = [true_hit.get_distance(hp - cb) for hp in self.good_hit_points]

            # min_dist = np.min(dist2hit)
            idx = np.argmin(dist2hit)
            hit_point = self.good_hit_points[idx]

            alpha = (true_hit - cb).normalized().dot((hit_point - cb).normalized())

    # min angle from target ball initial vel to a pocket
        if self.tracking["target_ball"] is not None:
            target_ball = self.tracking["target_ball"]
            target_pos = self.ball_tracking["start_positions"][target_ball.number]
            target_vel = self.ball_tracking["calc_velocity"][target_ball.number]

            pocket_vecs = [poc - target_pos for poc in self.target_points]
            min_angle = np.min([abs(target_vel.get_angle_degrees_between(x)) for x in pocket_vecs])

            self.rewardfunc = ['target_angle_pocket', min_angle]
            # print(f"Minimum angle: {min_angle}")

    # Window reward
        # if self.cue_alpha < 180:
        #     self.rewardfunc = ['window_cue', self.cue_alpha]
        # else:
        #     self.rewardfunc = ['window_cue', 180.0]

        # if self.target_alpha < np.inf:
        #     self.rewardfunc = ['window_target', self.target_alpha]


    # post calculation
        reward_pre = self.reward
        self.reward = np.clip(self.reward, -210, 210) / 210 # Normalization
        if not self.training:
            print('-'*50)
            print('Total Reward'.ljust(25),'{:0.2f}'.format(reward_pre))
            print('Total Reward (normalized): {:10.4f}'.format(self.reward))
            print('-'*50,end='\n\n')

    def plot_parms(self):
        x, y = self.cue_ball.body.position

        if self.winning_condition_met():
            self.win_list.append('r')
        else:
            self.win_list.append('b')

        t = [i[0] for i in self.actions]
        v = [i[1] for i in self.actions]
        plt.scatter(t, v, c=self.win_list)

        plt.show()

    def out_of_bounds(self):
        obs = self._get_obs()

        for i in range(0, len(obs[::3]), 3):
            if (obs[i] + 1e-2 < 0) or (obs[i] - 1e-2 > 1) or (obs[i + 1] + 1e-2 < 0) or (obs[i + 1] - 1e-2 > 1):
                return True

        return False

    def best_shot_criteria(self, best_vectors):
        hit_vectors, pocket_id, ball_number = zip(*best_vectors)
        scores = []
        self.scores = []

        for i in range(len(best_vectors)):

            b_id = ball_number[i] // 100
            p_id = self.pocket_ids[pocket_id[i]] // 100

            if b_id == 0 and p_id == 0:     # both in main table
                multiplier = 1
            elif b_id == p_id:              # cue cushion hit
                multiplier = 3/4
            elif b_id == 0 and p_id != 0:   # blue cushion hit
                multiplier = 1/2
            else:                           # different tables
                multiplier = 1/4

            cus1 = Vec2d(*self.cushion_corners[pocket_id[i]][0])
            cus2 = Vec2d(*self.cushion_corners[pocket_id[i]][1])

            if ball_number[i] < 100:
                ballpos = self.balls[ball_number[i]].body.position
            else:
                ballpos = self.ghost_balls[(ball_number[i] // 100) - 1]
            a1 = cus1 - ballpos
            a2 = cus2 - ballpos

            angle_between = abs(a1.get_angle_degrees_between(a2)) / 60

            # window
            self.window_vectors.append((ballpos,cus1,cus2,angle_between,pocket_id[i]))

            # Cosine_sim weight
            hit_vec = hit_vectors[i].normalized()
            poc_vec = (Vec2d(*self.target_points[pocket_id[i]]) - ballpos).normalized()

            cos_weight = hit_vec.dot(poc_vec)

            scores.append((angle_between + cos_weight) * multiplier)
            self.scores.append([angle_between,cos_weight,multiplier, scores[-1]])

        best = np.argmax(scores)
        bestscore = np.max(scores)
        return hit_vectors[best], best, bestscore

    def is_straight_line(self, main_pos, target_pos, exclude=[], include=[]):
        # Checks if there is an unobstructed line from main_pos to target_pos.
        # Main_pos: Vec2d
        # Target_pos: Vec2d
        # exclude: [Vec2d, Vec2d]   -- list of points to exclude other than main and target

        main_pos = Vec2d(*main_pos)
        target_pos = Vec2d(*target_pos)

        def point_on_line_seg(a, b, p):
            # a is start of line segment
            # b is end of line segment
            # p is point we want to find the distance to
            a = Vec2d(*a)
            b = Vec2d(*b)
            p = Vec2d(*p)
            ap = p - a
            ab = b - a

            t = ap.dot(ab) / ab.get_length_sqrd()
            # if you need the the closest point belonging to the segment
            t = max(0, min(1, t))
            point = a + (t * ab)
            dist = p.get_distance(point)
            return dist
    
        def hits(item, exlist,s,r=2):
            try: 
                [ball.body.position for ball in item]
            except:
                pos = np.array([Vec2d(*position) for position in item
                                      if Vec2d(*position) not in exlist]).reshape(-1, 2)
            else:
                pos = np.array([ball.body.position for ball in item 
                                      if ball.body.position not in exlist]).reshape(-1, 2)
            
            dists = [point_on_line_seg(exlist[0], exlist[1], ball) for ball in pos]
            hit = sum([abs(d) <= (r * cfg.BALL_RADIUS) for d in dists]) > 0
            return hit

        exlist = [main_pos, target_pos] + exclude        
        
        if hits(self.balls, exlist,"bal"):
            self.trash_lines.append([main_pos,target_pos,"ball"])
            return False
            
        elif hits(self.ghost_balls, exlist,"gho"):
            self.trash_lines.append([main_pos,target_pos,"ghost"])
            return False
            
        elif hits(self.ghost_opponents, exlist,"opp"):
            self.trash_lines.append([main_pos,target_pos,"ghost_opp"])
            return False
            
        elif hits(self.target_points, exlist,"poc"):
            self.trash_lines.append([main_pos,target_pos,"pocket"])
            return False

        return True  # The line has no obstructions
        
    
    def add_ghosts(self):
        self.ghost_balls = []
        self.ghost_opponents = []
        
        for ball in self.balls:
            real_ball_pos = ball.body.position

            ghosts = []
            ghost_opponents = []
            if self.bank_shots:
                for i, line in enumerate(cfg.CUSHION_INNER_LINES):
                    dist = real_ball_pos[int(i>=2)] - line
                    if dist > cfg.BALL_RADIUS or dist < -cfg.BALL_RADIUS:
                        if i < 2:   coord = (line-dist,real_ball_pos[1])
                        else:       coord = (real_ball_pos[0],line-dist)

                        if ball.ballclass == self.suit: ghosts.append(coord)
                        elif ball.ballclass != self.cue_ball.ballclass: ghost_opponents.append(coord)

            self.ghost_balls = self.ghost_balls + ghosts
            self.ghost_opponents = self.ghost_opponents + ghost_opponents
        
    
    def find_best_shot(self):
        # Find lines from balls to pockets
        self.good_vectors = []
        all_vectors = []
        self.draw_stuff["hit_points"] = []
        self.draw_stuff["hit_points_details"] = []
        self.best_shot = None
        self.hit_points = []
        self.good_hit_points = []

        cue_pos = self.cue_ball.body.position

        if self.num_balls == 1:
            random_int = np.random.randint(0, 6)
            pocket_pos = Vec2d(*self.target_points[random_int])
            self.best_shot = pocket_pos - cue_pos
            return None
        
        self.add_ghosts()

        for ball in self.balls:
            real_ball_pos = ball.body.position

            if ball.ballclass != self.suit:  # Only consider your own suit from here
                continue
            
            for ghostnum, ball_pos in enumerate([real_ball_pos] + self.ghost_balls):
                # Find good pockets
                for i, pocket in enumerate(self.target_points):
                    pocket_pos = Vec2d(*pocket)

                    # Calculate pos the cue should hit
                    pocket_vec = (pocket_pos - ball_pos).normalized()
                    hit_pos = ball_pos - ((2 - 0) * cfg.BALL_RADIUS * pocket_vec)

                    cue2hit_vector = (hit_pos - cue_pos).normalized()
                    theta = cue2hit_vector.get_angle_degrees_between(pocket_vec)

                    self.hit_points.append([Vec2d(*hit_pos), theta])  # Feasible hit_points
                    all_vectors.append([hit_pos - cue_pos, i, ghostnum*100 + ball.number])
                    if (theta > cfg.THETA_LIMIT) or (theta < -cfg.THETA_LIMIT):  # Bad pocket
                        continue


                    # Check if there exists a line from ball to pocket
                    if self.is_straight_line(ball_pos, pocket_pos,include=[hit_pos]):

                        # Check if there exists a line from cue to hit
                        if self.is_straight_line(cue_pos, hit_pos, exclude=[ball_pos]):

                            self.good_hit_points.append(Vec2d(*hit_pos))
                            self.draw_stuff["hit_points"].append(hit_pos)
                            self.draw_stuff["hit_points_details"].append([ball_pos,pocket_pos])
                            self.good_vectors.append([hit_pos - cue_pos, i, ghostnum*100 + ball.number])

        # ball for loop end
        if len(self.good_vectors) != 0:
            self.best_shot, best, self.best_score = self.best_shot_criteria(self.good_vectors)
            self.draw_stuff["hit_points_best"] = self.draw_stuff["hit_points"][best]
            self.draw_stuff["draw_hit_points"] = True

        else:
            self.best_shot, best, self.best_score = self.best_shot_criteria(all_vectors)
        return None

    def find_windows(self):
        # Find angle window from cue-ball to every ball of ones suit
        # self.window_cue_angles = [None * self.num_balls]
        cue_pos = self.ball_tracking["start_positions"][self.cue_ball.number]

        self.cue_alpha = 180
        for ball in self.balls:
            if ball.ballclass != self.suit:
                continue
            ball_pos = self.ball_tracking["start_positions"][ball.number]

            ball_vector = ball_pos - cue_pos
            ball_vector = 2 * cfg.BALL_RADIUS * ball_vector.normalized()
            w1 = (ball_pos + ball_vector.rotated_degrees(90 + 5) - cue_pos).angle_degrees
            w2 = (ball_pos + ball_vector.rotated_degrees(-90 - 5) - cue_pos).angle_degrees

            if (min(w1, w2) <= self.angle) and (self.angle <= max(w1, w2)) and self.is_straight_line(cue_pos, ball_pos):
                self.cue_alpha = 0
            else:
                self.cue_alpha = min([abs(self.angle - (w1+w1)/2), self.cue_alpha])

        # Find window from target ball to every pocket
        self.target_alpha = np.inf
        if self.tracking["target_ball"] is not None:
            target_ball = self.tracking["target_ball"]
            target_pos = self.ball_tracking["start_positions"][target_ball.number]
            target_angle = self.ball_tracking["init_velocity"][target_ball.number].angle_degrees
        else:
            return None

        for corners in cfg.CUSHION_CORNERS:
            corner1, corner2 = corners

            pocket_target = Vec2d(*np.array([corner1, corner2]).mean(axis=0))
            # Find vector from pocket_midpoint to corner
            r1 = Vec2d(*(corner1 - pocket_target)).normalized()
            r2 = Vec2d(*(corner2 - pocket_target)).normalized()

            # Find points that lie on the line between corners and is 1 ball radius away from actual corner
            r1 = corner1 - cfg.BALL_RADIUS * r1
            r2 = corner2 - cfg.BALL_RADIUS * r2

            w1 = Vec2d(*(r1 - target_pos)).angle_degrees
            w2 = Vec2d(*(r2 - target_pos)).angle_degrees

            if (min(w1, w2) <= target_angle) and (target_angle <= max(w1, w2)) and self.is_straight_line(target_pos, pocket_target):
                self.target_alpha = 0
            else:
                self.target_alpha = min([abs(target_angle - (w1+w2)/2), self.target_alpha])

        return None

    def valid_action_mask(self):
        # Return action mask
        angle_dim, velocity_dim = self.action_space.nvec
        mask = [False] * angle_dim + [True] * velocity_dim
        
        ball_pos = [ball.body.position for ball in self.balls if ball.ballclass == self.suit]
        
        # Go through all hit-points and ball centers and make those angles viable
        for vec in self.good_hit_points+ball_pos:
            hit_vec = vec - self.cue_ball.body.position
            valid_shot = int(np.round((hit_vec.angle_degrees + 180) * cfg.ANGLE_PRECISION))
            
            mask[valid_shot] = True
        
        # If somehow there are no "good" shots, select one at random.
        if sum(mask) == 0:
            random_idx = np.random.randint(0, len(mask))
            mask[random_idx] = True
            
        return mask

    def reset_tracking(self):
        self.tracking = {
            "is_success": False,
            "suit": self.suit,
            "cue_ball_pocketed": False,
            "black_ball_pocketed": False,
            "ball_pocketed": [],  # 1: stripe, 2: solid, 3: cue, 4: black
            "first_cue_contact": None,  # None (0), ball (1), pocket (2), cushion (3)
            "target_ball": None,  # None if no ball is hit, else is the first ball cue contacts.
            "first_suit_collision": None,
            "cushion_hit": [],  #
            "ball_contacts": [],
            "contact_point": [],
            "contact_point_dist": [],
            "ball_collision": False,
            "last_cushion_vel": Vec2d.zero(),
            "normal_vectors": [],
            "last_ball_contact_vel": [Vec2d.zero() for _ in range(self.num_balls)],
            "blue_in": False,
            "blue_and_cue_in": False
        }
        self.ball_tracking = {
            "start_positions": [ball.body.position for ball in self.balls],
            "ball_classes": [ball.ballclass for ball in self.balls],
            "init_velocity": [Vec2d.zero() for _ in range(self.num_balls)],
            "calc_velocity": [Vec2d.zero() for _ in range(self.num_balls)],
            "cue_pos": [],
            "ball_pos": [[] for _ in range(self.num_balls)]
        }

        self.ball_collision_handler.data["cue_ball"] = self.cue_ball
        self.ball_collision_handler.data["tracking"] = self.tracking
        self.ball_collision_handler.data["ball_tracking"] = self.ball_tracking
        self.pocket_collision_handler.data["balls"] = self.balls
        self.pocket_collision_handler.data["suit"] = self.suit
        self.pocket_collision_handler.data["tracking"] = self.tracking
        self.cushion_collision_handler.data["tracking"] = self.tracking

        self.trash_lines = []

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Remove balls, pockets and cushion
        for x in self.space.shapes:
            try:
                self.space.remove(x)
                self.space.remove(x.body)
            except AssertionError:
                pass

        # Remove joints (those purple dots on the balls)
        for x in self.space.constraints:
            try:
                self.space.remove(x)
            except AssertionError:
                pass

        self.steps_taken = 0
        self.reward = 0
        self.lost = False
        self.win = False
        self.window_vectors = []
        self.prev_pocketed_balls = []
        
        # Add balls and table
        self.add_table()
        self.add_balls()
        
        # Change black to your suit when all your balls are pocketed
        if sum([ball.ballclass == self.suit for ball in self.balls]) == 0:
            for ball in self.balls:
                if ball.ballclass == 4:
                    ball.ballclass = self.suit
                    
        self.reset_tracking()
        self.find_best_shot()
        self.draw_stuff["draw_hit_points"] = False
        
        self.drawing_state = "start"
        if self.obs_type == "image" or not self.training:
            self.render()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.steps_taken += 1
        self.window_vectors = []
        self.fakehits = []

        self.find_best_shot()
        self.n_hp.append(len(self.good_hit_points))
        
        self.number_hit_point = len(self.good_hit_points)
        if not self.replay_state:
            self.current_hit_point = 0
            self.hit_success.append([False]*self.number_hit_point)

        self.actions.append(action)
        self.angle, self.force = action


        if self.algorithm in ["TD3", "DDPG", "SAC"]:
            if self.oracle: self.force = 1

            self.angle *= cfg.N_ANGLES / (2 * cfg.ANGLE_PRECISION)  # angle maps from [-1, 1] -> [-180, 180]
            self.force = (175 * self.force + 5) * cfg.ZOOM  # Force maps from [0, 1] -> [5, 180] * ZOOM

        else:
            if self.oracle: self.force = cfg.FORCE-1

            self.angle = self.angle / cfg.ANGLE_PRECISION - 180
            self.force = (245/(cfg.FORCE-1) * self.force + 5) * cfg.ZOOM # Force will be mapped from [0, FORCE-1] -> [5, 180] * ZOOM


        if self.oracle:
            self.angle = self.best_shot.angle_degrees

        if self.sigma != 0.0:
            self.angle += np.random.normal(loc=0, scale=self.sigma)

        self.reward = 0

        self.ball_collision_handler.data["tracking"]["first_cue_contact"] = None
        self.pocket_collision_handler.data["tracking"]["first_cue_contact"] = None
        self.cushion_collision_handler.data["tracking"]["first_cue_contact"] = None

        x_impulse = np.cos(np.radians(self.angle))
        y_impulse = np.sin(np.radians(self.angle))
        self.cue_shot_vec = Vec2d(self.force * x_impulse, self.force * y_impulse)

        self.cue_ball.body.activate()
        pymunk.Body.update_velocity(self.cue_ball.body, self.cue_shot_vec, damping=0, dt=1)

        self.ball_tracking["init_velocity"][self.cue_ball.number] = self.cue_ball.body.velocity
        self.ball_tracking["calc_velocity"][self.cue_ball.number] = self.cue_ball.body.velocity

        self.terminated = False
        self.truncated = False
        info = {}

        # Calculate best shot
        self.drawing_state = "end"
        frame_counter = 0
        while self.running:

            frame_counter += 1
            # Check if balls have stopped moving
            for ball in self.balls:
                if not ball.body.is_sleeping:
                    break
            else:
                break

            # === SIMULATION === #
            if (frame_counter % 10_000) == 0:
                print("HELP IM STUCK IN THE WHILE LOOP")
                print(f"State: {self.random_state}")
                ball_p = [ball.body.position for ball in self.balls]
                ball_v = [ball.body.velocity for ball in self.balls]
                print(f"Balls current position: {ball_p}")
                print(f"Balls current velocity: {ball_v}")

                print()
                initial_pos = self.ball_tracking["start_positions"]
                print(f"Inital positions: {initial_pos}")
                print(f"Action taken: ({self.angle}, {self.force})")
                break

            # step through
            if not self.training:
                for ball in self.balls:
                    self.ball_tracking["ball_pos"][ball.number].append(ball.body.position)

                if frame_counter % self.frame_rate == 0:
                    self.render()

                    while self.paused:
                        self.process_events()
                        self.clock.tick(15)


                for _ in range(self.dt):
                    self.space.step(self.step_size / self.dt)
                # for _ in range(self.dt):
                #     self.space.step(self.step_size / self.dt)
            else:
                for _ in range(self.dt):
                    self.space.step(self.step_size / self.dt)



        # ========== WHILE LOOP END ========== #

        self.no_suits_left = (sum([ball.ballclass == self.suit for ball in self.balls]) == 0)
        if self.num_balls >= 2:
            if self.no_suits_left and (not self.tracking["cue_ball_pocketed"]):
                # If all your balls have been pocketed (+ black, since it changes to your suit in the end), you win
                self.win = True
                self.terminated = True
                self.tracking["is_success"] = True
    
            elif (len(self.tracking["ball_pocketed"]) != 0):
                pocketed_classes = [ball.ballclass for ball in self.tracking["ball_pocketed"]]
                # 4: black, 3: cue, 2: solid, 1: stripe
    
                if 4 in pocketed_classes:
                    # Black was pocketed -> you loose the game
                    self.terminated = True
                    self.lost = True
                elif 3 in pocketed_classes:
                    # Cue was pocketed -> you miss your turn
                    self.terminated = True
                    if self.no_suits_left:
                        # If no suits left and cue ball pocketed -> you loose
                        self.lost = True
    
                elif self.enemy_suit in pocketed_classes:
                    # Enemy suit was pocketed -> you miss your turn
                    self.terminated = True
                
                else:
                    # Only your suit was pocketed -> you get a turn
                    self.terminated = False
                    self.steps_taken = 0
    
            else:
                # No balls were pocketed -> you miss your turn if number of steps exceed maximum allowed number of shots
                if self.steps_taken >= self.max_num_shots:
                    self.terminated = True
                else:
                    self.terminated = False
        
        else:
            if self.tracking["cue_ball_pocketed"]:
                self.win = True
                self.terminated = True
                self.tracking["is_success"] = True

        if len(self.balls) < 2:
            self.terminated = True

        if self.obs_type == "vector":
            self.truncated = self.out_of_bounds()

        self.drawing_state = "end"
        if self.obs_type == "image" or not self.training:
            self.render()

        self.observation = self._get_obs()

        self._get_reward()
        self.prev_reward = self.reward
        self.stop_time = time.perf_counter()

        if (len(self.good_hit_points) == 0) and (not self.win) and (not self.tracking["cue_ball_pocketed"]):
            self.no_hitpoint_counter += 1

        self.cue_pocketed += self.tracking["cue_ball_pocketed"] and self.no_suits_left

        if ((self.total_steps % 1000) == 0) and (self.total_steps != 0) and (self.truncated or self.terminated):
            timer = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
            print(f"\nTotal steps: \t{self.total_steps}\t Time: {timer}")
            print("Wins:     {0:5.0f}\t {1:5.2f}% \t({2:5.0f}/1000\t {3:5.2f}%)".format(self.wins, self.wins / self.total_steps * 100, self.win_pr_1000, (self.win_pr_1000) / 10))
            print("Hitpoint: {0:<5}\t {0:<5} \t\t({2:5.0f}/1000\t {3:5.2f}%)".format('', '', self.no_hitpoint_counter, (self.no_hitpoint_counter) / 10))
            print("Cue:      {0:<5}\t {0:<5} \t\t({2:5.0f}/1000\t {3:5.2f}%)".format('', '', self.cue_pocketed, (self.cue_pocketed) / 10))
            print("Remaining:{0:<5}\t {0:<5} \t\t({2:<9}\t {3:5.2f}%)".format('', '', '', 100 - (self.cue_pocketed + self.no_hitpoint_counter + self.win_pr_1000) / 10))

            self.win_pr_1000 = 0
            self.no_hitpoint_counter = 0
            self.cue_pocketed = 0

        if self.truncated:
            print(f"Truncation has occured at state: {self.random_state}")
            # self.pretty_print_dict(self.get_attrs())

        # Shift ball numbers down
        for ball in self.tracking["ball_pocketed"]:
            number = ball.number
            for b in self.balls:
                if b.number >= number:
                    b.number = b.number - 1

        
        for ball in self.tracking["ball_pocketed"]:
            self.prev_pocketed_balls.append(ball.ballclass)
        
        self.tracking["blue_in"] = self.no_suits_left
        self.tracking["blue_and_cue_in"] = self.no_suits_left and self.tracking["cue_ball_pocketed"]
        info = self._get_info()
        
        self.reset_tracking()
        if self.terminated or self.truncated:
            self.total_steps += 1

        return self.observation, self.reward, self.terminated, self.truncated, info

    def pretty_print_dict(self, d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                self.pretty_print_dict(value, indent + 1)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self.pretty_print_dict(item, indent + 1)
                    else:
                        print('\t' * (indent + 1) + str(item))
            else:
                print('\t' * (indent + 1) + str(value))
        print('\n')

    def process_events(self):
        for event in pygame.event.get():
            # Quit sim
            if (event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and event.key == pygame.K_q):
                self.running = False
                self.close()

            # Print attr
            if (event.type == pygame.KEYDOWN) and (event.key == pygame.K_p):
                print("Pressed P")
                # print(self.cue_ball.body.position)
                self.pretty_print_dict(self.get_attrs())

            # view lines trigger
            if (event.type == pygame.KEYDOWN) and (event.key == pygame.K_l) and (not self.training):
                print("Pressed L")
                self.view_lines = not self.view_lines

            # Replay state
            if (event.type == pygame.KEYDOWN) and (event.key == pygame.K_r) and (not self.training):
                print("Pressed R")
                self.Continue = self.replay_state
                self.replay_state = not self.replay_state

                print(f"Replay is set to {self.replay_state}")

            # Pause
            if (event.type == pygame.KEYDOWN) and (event.key == pygame.K_SPACE) and (not self.training):
                print("Pressed SPACE")
                self.paused = not self.paused

    def draw_state(self):
        # Draw hit points
        bestline = None
        for num,pos in enumerate(self.draw_stuff["hit_points"]):


            if pos == self.draw_stuff["hit_points_best"]:
                color = cfg.BESTSHOTCOLOR
            else:
                color = cfg.SHOTCOLOR

            start = self.ball_tracking["start_positions"][self.cue_ball.number]
            end = pos
            if pos == self.draw_stuff["hit_points_best"]:
                bestline = [start,
                            end,
                            self.draw_stuff["hit_points_details"][num][0],
                            self.draw_stuff["hit_points_details"][num][1]]

            if self.view_lines:
                pygame.draw.circle(self.screen, color, pos, 4 * cfg.ZOOM)
                pygame.draw.line(self.screen, color,
                                  start, end, width=int(2 * cfg.ZOOM))
                pygame.draw.line(self.screen, color,
                                  self.draw_stuff["hit_points_details"][num][0],
                                  self.draw_stuff["hit_points_details"][num][1], width=int(2 * cfg.ZOOM))

        if self.view_lines:
            font = pygame.font.SysFont('didot.ttc', 20)

            self.screen.blit(font.render("Best Shot", True, cfg.BESTSHOTCOLOR), (100,100))
            self.screen.blit(font.render("Possible Shots", True, cfg.SHOTCOLOR), (100,120))
            self.screen.blit(font.render("Interupted Shots", True, cfg.INTERUPTEDCOLOR), (100,140))

        # Trash lines
            for i,line in enumerate(self.trash_lines):
                if line[2] == "ball":
                    trashcol = cfg.RED
                elif line[2] == "ghost":
                    trashcol = cfg.GRAY
                elif line[2] == "ghost_opp":
                    trashcol =cfg. MAGENTA
                elif line[2] == "pocket":
                    trashcol = cfg.CYAN
                elif line[2] == "cushion":
                    trashcol = cfg.PURPLE
                pygame.draw.line(self.screen, trashcol,
                                  line[0], line[1], width=int(2 * cfg.ZOOM))


        txts = [[0] for _ in self.target_points]

        if cfg.VIEW_MIRRORS:
            for i,vecset in enumerate(self.window_vectors):
                font = pygame.font.SysFont('didot.ttc', 28)

                if len(self.scores) == len(self.window_vectors):
                    prod_scores = self.scores[i][-1]

                    if prod_scores == self.best_score:
                        text = font.render("{0:.2f}".format(prod_scores), True, cfg.BESTSHOTCOLOR)
                    else:
                        text = font.render("{0:.2f}".format(prod_scores), True, cfg.BLACK)
                    txts[vecset[4]].append([text,vecset[1]])

        for pock in txts:
            for i,txt in enumerate(pock):
                if txt!=0: self.screen.blit(txt[0], (txt[1][0]+20,txt[1][1]+(i*20)))

        if self.replay_state:
            font = pygame.font.SysFont('didot.ttc', 40)
            text = font.render("Replay", True, cfg.RED)
            self.screen.blit(text, (cfg.LOWER_X,20))
            
        if self.Continue:
            font = pygame.font.SysFont('didot.ttc', 40)
            text = font.render("Continue", True, cfg.RED)
            self.screen.blit(text, (cfg.LOWER_X,20))


        if self.drawing_state == "end":
            # Draw cue shot
            vec = self.ball_tracking["init_velocity"][self.cue_ball.number]
            start = self.ball_tracking["start_positions"][self.cue_ball.number]
            end = start + vec * cfg.ZOOM * 2
            
        if self.bank_shots:
            for i in range(4):
                if i < 2: pygame.draw.line(self.screen, cfg.TABLE_SIDE_COLOR, (cfg.CUSHION_INNER_LINES[i],0), (cfg.CUSHION_INNER_LINES[i], cfg.FULL_SCREEN_HEIGHT))
                else: pygame.draw.line(self.screen, cfg.TABLE_SIDE_COLOR, (0, cfg.CUSHION_INNER_LINES[i]), (cfg.FULL_SCREEN_WIDTH, cfg.CUSHION_INNER_LINES[i]))

            if cfg.VIEW_MIRRORS:
                for pos in self.ghost_balls:
                    pygame.draw.circle(self.screen, cfg.YELLOW, pos, cfg.BALL_RADIUS)

                for pos in self.ghost_opponents:
                    pygame.draw.circle(self.screen, cfg.MAGENTA, pos, cfg.BALL_RADIUS)


        # Draw collision points
        for cp_pair, dist in zip(self.tracking["contact_point"], self.tracking["contact_point_dist"]):
            for i, pos in enumerate(cp_pair):
                if i == 0:
                    font = pygame.font.SysFont('didot.ttc', 20)
                    text = font.render(str(round(dist, 2)), True, cfg.RED)

        # Track balls position
        for i in range(len(self.ball_tracking["ball_pos"])):

            for pos in self.ball_tracking["ball_pos"][i]:
                if self.ball_tracking["ball_classes"][i] == self.suit:
                    pygame.draw.circle(self.screen, cfg.BLUE, pos, 3)
                elif self.ball_tracking["ball_classes"][i] == 1:
                    pygame.draw.circle(self.screen, cfg.SOLID_COLOR, pos, 3)
                elif self.ball_tracking["ball_classes"][i] == 3:
                    pygame.draw.circle(self.screen, cfg.WHITE, pos, 3)


        for i in range(len(self.ball_tracking["start_positions"])):
            start = self.ball_tracking["start_positions"][i]  # Vec2d
            end = start + self.ball_tracking["init_velocity"][i] * 2  # Vec2d
            end2 = start + self.ball_tracking["calc_velocity"][i] # Vec2d
            ball_class = self.ball_tracking["ball_classes"][i]

            pygame.draw.circle(self.screen, cfg.SUIT_COLORS[ball_class - 1], start, cfg.BALL_RADIUS*1.3, width=int(1 * cfg.ZOOM))

        
        # draw fake hitpoints
        for h in self.fakehits:
            pygame.draw.line(self.screen, cfg.CYAN, h[0],h[1], width=int(1 * cfg.ZOOM))

        #draw best line on top
        # bestline = None
        if bestline is not None:
            path1 = (bestline[0],bestline[1])
            path2 = (bestline[2],bestline[3])

            if self.bank_shots:
                for i, line in enumerate(cfg.CUSHION_INNER_LINES):
                    y = int(i>1)
                    
                    for path in [path1,path2]:
                        dist1 = path[0][y] - line   # distance from x1 to line
                        dist2 = path[1][y] - line   # distance from x2 to line
                        
                        show = sum([[c[y]<line, c[y]>line, c[y]>line, c[y]<line][i] for c in path])!=0
                        
        # display pocketed balls
        blues = 0
        for i,ball in enumerate(self.prev_pocketed_balls + [ball.ballclass for ball in self.tracking["ball_pocketed"]]):
            blues += int(ball==2)
            if ball==2 and blues > self.numblues: pygame.draw.circle(self.screen, cfg.BLACK, (cfg.LOWER_X+(3*cfg.BALL_RADIUS)*(i+1),cfg.UPPER_Y+(5*cfg.BALL_RADIUS)), cfg.BALL_RADIUS)
            else: pygame.draw.circle(self.screen, cfg.SUIT_COLOR[ball], (cfg.LOWER_X+(3*cfg.BALL_RADIUS)*(i+1),cfg.UPPER_Y+(5*cfg.BALL_RADIUS)), cfg.BALL_RADIUS)
                

    def draw_vectors(self):
        # Draw vector from cue to best hit and cue shot vector
        if self.draw_stuff["draw_hit_points"]:
            # Best hit vector
            start = self.ball_tracking["start_positions"][self.cue_ball.number]  # Vec2d
            end = self.draw_stuff["hit_points_best"]  # Tuple
            # pygame.draw.line(self.screen, CYAN,
            #                  start, end, width=int(3 * ZOOM))

        # Draw collision points
        if self.tracking["ball_collision"]:
            for pos in self.tracking["contact_point"]:
                self.screen.fill(cfg.BLUE, (pos, (5 * cfg.ZOOM, 5 * cfg.ZOOM)))
                pygame.draw.circle(self.screen, cfg.YELLOW, pos, 2*cfg.ZOOM)

        # Draw circle around balls + initial velocity
        for i in range(len(self.ball_tracking["start_positions"])):
            start = self.ball_tracking["start_positions"][i]  # Vec2d
            end = start + self.ball_tracking["init_velocity"][i] * 2  # Vec2d
            end2 = start + self.ball_tracking["calc_velocity"][i] # Vec2d
            ball_class = self.ball_tracking["ball_classes"][i]

            pygame.draw.line(self.screen, cfg.DRAW_SHOT_COLOR[ball_class - 1],
                              start, end, width=int(3 * cfg.ZOOM))

            if i != self.cue_ball.number:
                pygame.draw.line(self.screen, cfg.BLACK,
                                  start, end2, width=int(2 * cfg.ZOOM))

            # self.screen.fill(SUIT_COLORS[ball_class - 1], (start, (10*ZOOM,10*ZOOM)))
            pygame.draw.circle(self.screen, cfg.SUIT_COLORS[ball_class - 1], start, cfg.BALL_RADIUS*1.3, width=int(1*cfg.ZOOM))


    def redraw_screen(self):
        self.screen.fill(pygame.Color(cfg.BG_COLOR))

        pygame.draw.polygon(self.screen, cfg.TABLE_COLOR, cfg.TABLE_SPACE)

        self.draw_state()
        self.space.debug_draw(self.draw_options)

        sp = " "*5

        captionMAIN = f"Pool {sp} FPS: {self.fps} {sp} Algorithm: {self.algorithm}" + sp
        captionSUIT = f"Your suit: {cfg.SUIT_NAMES[self.suit-1]} ({cfg.SUIT_COLORS[self.suit-1]})" + sp
        captionREWARD = f"previous reward: {round(self.prev_reward,3)} " + sp
        captionSTATE= f"random_state_nr: {self.random_state}" + sp
        if self.best_score: captionVALUE = f"HitPoint Value: {round(self.best_score,2)}" + sp
        else: captionVALUE = f"HitPoint Value: 0" + sp
        caption = captionMAIN + captionSUIT + captionSTATE + captionREWARD + captionVALUE
        pygame.display.set_caption(caption)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def render(self):
        self.redraw_screen()
        self.process_events()

    def get_attrs(self):
        def get_balls(x):
            return {
                "position": list(x.body.position),
                "velocity": list(x.body.velocity),
                "number": x.number,
                "color": x.color,
                # "filter": x.filter,
                "collision_type": x.collision_type,
                "ball_class": x.ballclass,
            }

        return {
            "balls_attrs": [get_balls(x) for x in self.balls],
            "reward": copy(self.reward),
            "total_steps": copy(self.total_steps),
            "starting_time": copy(self.start_time),
            # "ball_tracking": copy(self.score_tracking),
            "tracking": copy(self.tracking),
            "ball_tracking": copy(self.ball_tracking),
            "won": copy(self.win),
            "terminated": copy(self.terminated),
            "truncated": copy(self.truncated)
        }

    def close(self):
        print('CALLED: close()')
        pygame.display.quit()
        pygame.quit()
        sys.exit()
