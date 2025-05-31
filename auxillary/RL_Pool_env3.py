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
from RL_config_env import *
import matplotlib.pyplot as plt
import time
from copy import copy


class PoolEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
            self,
            algo="PPO",
            balls_init=None,
            suit=2,  # 1: Solid, 2: Stripe
            training=True,
            fps=60,
            num_balls=2,
            special_state=None,
            obs_type="vector",
            sigma=0,
            cheat_force = False,
            cheat_angle = False,
            game_type = 'blue_only',
            bank_shots = True,
    ):
        #
        self.algorithm = algo
        self.balls_init = balls_init
        self.suit = suit
        self.training = training
        self.fps = fps
        self.obs_type = obs_type
        self.sigma = sigma
        self.cheat_force = cheat_force
        self.cheat_angle = cheat_angle
        self.game_type = game_type

        self.bank_shots = bank_shots #  add extra target pockets
        self.balls = []
        self.ghost_balls = []
        self.ghost_opponents = []
        self.fakehits = []
        self.scores = []
        self.best_score = 0

        self.width = FULL_SCREEN_WIDTH
        self.height = FULL_SCREEN_HEIGHT
        self.num_balls = num_balls if balls_init is None else self.balls_init.shape[0]
        self.total_balls = 16
        self.draw_screen = (self.obs_type == "image") or not training  # Replace with some logic to determine when to draw the screen.
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
            self.action_space = spaces.MultiDiscrete([N_ANGLES, FORCE])
            # self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), shape=(2,))

        self.actions = [[0,0]] 

        if self.obs_type == "vector":
            box_low = np.array([0.0, 0.0, 0] * self.total_balls)
            # box_high = np.array([self.width, self.height, 1] * self.total_balls)
            box_high = np.array([1.0, 1.0, 1] * self.total_balls)
            # box_high = np.array([1,1,1]*self.num_balls)
            # box_shape = (1,3 * num_balls)
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
        self.space.damping = BALL_DAMPING_THRESHOLD  # 0.83
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
        self.test_all_hp = False
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
            for i,line in enumerate(CUSHION_INNER_LINES):
                for k,target in enumerate(TARGET_POSITIONS):
                    dist = target[int(i>=2)] - line
                    dist1 = CUSHION_CORNERS[k][0][int(i>=2)] - line
                    dist2 = CUSHION_CORNERS[k][1][int(i>=2)] - line
                    if dist > POCKET_RADIUS or dist < -POCKET_RADIUS:
                        self.pocket_ids.append(100*(i+1) + k)
                        if i < 2:
                            extra_targets.append([line-dist,target[1]])
                            extra_corners.append([[line-dist1,CUSHION_CORNERS[k][0][1]],[line-dist2,CUSHION_CORNERS[k][1][1]]])
                        else:
                            extra_targets.append([target[0],line-dist])
                            extra_corners.append([[CUSHION_CORNERS[k][0][0],line-dist1],[CUSHION_CORNERS[k][1][0],line-dist2]])


            self.target_points = np.concatenate([TARGET_POSITIONS,extra_targets])
            self.cushion_corners = np.concatenate([CUSHION_CORNERS,extra_corners])

        else:
            self.target_points = np.array([list(x) for x in TARGET_POSITIONS])
            self.cushion_corners = np.array(CUSHION_CORNERS)


        # print(self.target_points)
        self.window_vectors = []
        self.pocket_value_counter = np.zeros_like(self.target_points[:,0])
        # for i,c in enumerate(self.target_points):
        #     plt.scatter(c[0],c[1],color='b',s=2)
        #     plt.text(c[0]+50, c[1], str(i))

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

    # @staticmethod
    # def ball_post_solve(arbiter, space, data):
    #     ball1, ball2 = arbiter.shapes

        # print()
        # print("Post_solve")
        # print(f"Ball1: {ball1.body.velocity}")
        # print(f"Ball2: {ball2.body.velocity}")

    @staticmethod
    def ball_post_solve(arbiter, space, data):

        ball1, ball2 = arbiter.shapes
        point_a = arbiter.contact_point_set.points[0].point_a  # collision point on ball1
        point_b = arbiter.contact_point_set.points[0].point_b  # collision point on ball2

        # print("Velocities post_solve:")
        # print(f"Ball {ball1.ballclass}: {ball1.body.velocity}")
        # print(f"Ball {ball2.ballclass}: {ball2.body.velocity}")

        speed = ball1.body.velocity.length
        old_vel = ball1.body.velocity.normalized()
        # new_vel = (point_b - point_a).normalized()
        # new_vel = (point_b - ball2.body.position).normalized()
        # new_vel = -arbiter.normal.normalized()
        # new_vel = (ball1.body.position - ball2.body.position).normalized()

        v1 = data["tracking"]["last_ball_contact_vel"][ball1.number]
        v2 = data["tracking"]["last_ball_contact_vel"][ball2.number]
        x1 = ball1.body.position
        x2 = ball2.body.position
        # print()
        # print()
        # print(f"|x1-x2|: {(x1-x2)}")
        # print(((v1 - v2).dot((x1 - x2)) / (x1 - x2).length**2) * (x1 - x2))

        # https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional

        dv1 = ((v1 - v2).dot((x1 - x2)) / (x1 - x2).length**2) * (x1 - x2)
        dv2 = ((v2 - v1).dot((x2 - x1)) / (x2 - x1).length**2) * (x2 - x1)

        # k = 20
        # if dv1.length < k:
        #     dv1 = dv1.normalized() * k
        #     dv2 = dv2.normalized() * k
        # print(dv1.length)
        # print(dv2.length)
        new_vel1 = v1 - dv1
        new_vel2 = v2 - dv2
        # new_vel1
        if data["ball_tracking"]["init_velocity"][ball1.number] == Vec2d.zero():
            # data["ball_tracking"]["init_velocity"][ball1.number] = ball1.body.velocity
            data["ball_tracking"]["init_velocity"][ball1.number] = ball1.body.velocity
            data["ball_tracking"]["calc_velocity"][ball1.number] = new_vel1
        # ball1.body.velocity = (new_vel + old_vel).normalized() * speed
        ball1.body.velocity = new_vel1

        # if ball1.number == data["cue_ball"].number:
        #     data["cue_ball"].body.velocity = new_vel

        speed = ball2.body.velocity.length
        old_vel = ball2.body.velocity.normalized()
        # new_vel = (point_a - point_b).normalized()
        # new_vel = (point_a - ball1.body.position).normalized()
        # new_vel = -arbiter.normal.normalized().perpendicular()
        # new_vel = (ball2.body.position - ball1.body.position).normalized()
        if data["ball_tracking"]["init_velocity"][ball2.number] == Vec2d.zero():
            # data["ball_tracking"]["init_velocity"][ball2.number] = ball2.body.velocity
            data["ball_tracking"]["init_velocity"][ball2.number] = ball2.body.velocity
            data["ball_tracking"]["calc_velocity"][ball2.number] = new_vel2
        # ball2.body.velocity = (new_vel + old_vel).normalized() * speed
        ball2.body.velocity = new_vel2
        # if ball2.number == data["cue_ball"].number:
        #     data["cue_ball"].body.velocity = new_vel
        data["tracking"]["normal_vectors"].append(arbiter.normal)
        # print(f"Friction: {arbiter.friction}")
        # print(f"Elasticity: {arbiter.restitution}")
        # print(f"KE lost: {arbiter.total_ke}")

        # print(f"Ball1 vel (pymunk): {ball1.body.velocity}")
        # print(f"Ball1 vel (us):     {new_vel1}")

        # print(f"Ball2 vel (pymunk): {ball2.body.velocity}")
        # print(f"Ball2 vel (us):     {new_vel2}")
        # print()
        # print()
        return True


    @staticmethod
    def ball_contacted(arbiter, space, data):
        # count bank/carrom shot collisions
        # print('Ball contacted!!')
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

        # if data["ball_tracking"]["init_velocity"][ball1.number] == Vec2d.zero():
        #     vel = (ball1.body.position - ball2.body.position).normalized()
        #     data["ball_tracking"]["init_velocity"][ball1.number] = vel * 200

        # if data["ball_tracking"]["init_velocity"][ball2.number] == Vec2d.zero():
        #     vel = (ball2.body.position - ball1.body.position).normalized()
        #     data["ball_tracking"]["init_velocity"][ball2.number] = vel * 200

        for contact_point in arbiter.contact_point_set.points:
            data["tracking"]["contact_point"].append((contact_point.point_a, contact_point.point_b))
            data["tracking"]["contact_point_dist"].append(contact_point.distance)
            # data["tracking"]["contact_point"].append(contact_point.point_b)

        data["tracking"]["last_ball_contact_vel"][ball1.number] = ball1.body.velocity
        data["tracking"]["last_ball_contact_vel"][ball2.number] = ball2.body.velocity

        # print("Velocities contact:")
        # print(f"Ball {ball1.ballclass}: {ball1.body.velocity}")
        # print(f"Ball {ball2.ballclass}: {ball2.body.velocity}")
        # print(f"|x1-x2|: {(ball1.body.position-ball2.body.position)}")
        # arbiter.total_ke = 0
        # print(arbiter.contact_point_set.points[0].distance)
        # print(arbiter.contact_point_set.points[1].distance)
        # print(data["tracking"]["contact_point"])
        # Update s_ball shot_vec
        # vec = np.array(s_ball.body.position) - np.array(c_ball.body.position)
        # data["ball_tracking"]["shot_vectors"][s_ball.number] = vec / (np.linalg.norm(vec) + 1e-8)
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

        # number = ball.number
        try:
            data["balls"].remove(ball)
        except ValueError:
            pass
        space.remove(ball, ball.body)

        # # Shift ball numbers down
        # for b in data["balls"]:
        #     if b.number > number:
        #         b.number = b.number - 1

        # Implement note: change black to your suit when all your balls are pocketed
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
            # data["tracking"]["contact_point"].append(contact_point.point_b)
        # print(ball.body.velocity)
        d = ball.body.velocity
        data["tracking"]["last_cushion_vel"] = ball.body.velocity
        # ball.body.velocity = Vec2d.zero()
        # ball.body.velocity *= 0.1
        return True

    @staticmethod
    def cushion_post_solve(arbiter, space, data):
        ball, rail = arbiter.shapes
        # print(ball.body.velocity)
        speed = ball.body.velocity.length
        d = data["tracking"]["last_cushion_vel"]
        # print(arbiter.normal)
        n = arbiter.normal
        # data["tracking"]["normal_vectors"].append(arbiter.normal)
        new_vel = (d - 2 * (d.dot(n)) * n).normalized() * speed
        ball.body.velocity = new_vel

        # print(new_vel)


    def add_balls(self):
        
        self.balls = []
        positions = []
        amount_solids = int(np.floor((self.num_balls - 2) / 2))
        amount_stripes = int(np.ceil((self.num_balls - 2) / 2))
        
        self.numblues = amount_stripes
        
        b_number = 0

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

            # self.random_state = np.random.randint(0, self.balls_init.shape[2])

            # self.random_state = FAILED_STATES[self.total_steps % len(FAILED_STATES)]

        for i in range(self.num_balls):
            intertia = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS, offset=(0, 0))
            # intertia = np.inf
            ball_body = pymunk.Body(BALL_MASS, intertia)

            # initialize positions
            if self.balls_init is not None:
                if len(self.balls_init.shape) == 2:
                    x, y = self.balls_init[i, :2]
                    ball_body.position = [int((UPPER_X - LOWER_X - 2)*x + LOWER_X) + 1,
                                          int((UPPER_Y - LOWER_Y - 2)*y + LOWER_Y) + 1
                                          ]
                    ballclass = int(self.balls_init[i, 2])

                # Below is used for test purposes
                elif len(self.balls_init.shape) == 3:
                    x, y = self.balls_init[i, :2, self.random_state]
                    ball_body.position = [int((UPPER_X - LOWER_X - 2)*x + LOWER_X) + 1,
                                          int((UPPER_Y - LOWER_Y - 2)*y + LOWER_Y) + 1
                                          ]
                    ballclass = int(self.balls_init[i, 2, self.random_state])

            else:
                # Random position
                new_ball_x = np.random.randint(LOWER_X, UPPER_X)
                new_ball_y = np.random.randint(LOWER_Y, UPPER_Y)
                overlap = True

                while positions and overlap:  # While other balls and overlap
                    overlap = False

                    for ball in positions:
                        # Calculate distance to every other ball
                        dist = np.sqrt((abs(new_ball_x - ball[0]))**2 + (abs(new_ball_y - ball[1]))**2)

                        if dist <= 2* BALL_RADIUS:
                            # If overlap, try again
                            new_ball_x = np.random.randint(LOWER_X, UPPER_X)
                            new_ball_y = np.random.randint(LOWER_Y, UPPER_Y)
                            overlap = True
                            break
                # if no overlap, new ball position is valid
                ball_body.position = [new_ball_x, new_ball_y]
                ballclass = ball_classes[i]
                # ball_body.ballclass = ballclass

            ball = pymunk.Circle(ball_body, BALL_RADIUS, offset=(0, 0))
            ball.elasticity = BALL_ELASTICITY
            ball.friction = BALL_FRICTION
            ball.collision_type = 1
            ball.ballclass = ballclass

            ball.color = pygame.Color(SUIT_COLORS[ballclass-1])
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
        for cushion_pos in CUSHION_POSITIONS:
            cushion = pymunk.Poly(static_body, cushion_pos)
            cushion.color = pygame.Color(TABLE_SIDE_COLOR)
            cushion.collision_type = 3
            cushion.elasticity = CUSHION_ELASTICITY
            cushion.friction = CUSHION_FRICTION
            self.cushions.append(cushion)


        self.pockets = []
        for pocket_loc in POCKET_CENTERS:
            pocket = pymunk.Circle(static_body, POCKET_RADIUS, pocket_loc)
            pocket.color = pygame.Color(BLACK)
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
            obs = np.array([[(ball.body.position[0] - LOWER_X) / (UPPER_X - LOWER_X),
                             (ball.body.position[1] - LOWER_Y) / (UPPER_Y - LOWER_Y),
                             ball.ballclass / 4] for ball in self.balls])

            balls_to_fill = self.total_balls - len(self.balls)
            if len(self.balls) == 0:  # PROBLEM WITH THIS (maybe not?)
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
                    normdist = dist / DIAGONAL
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
                normdist = dist / DIAGONAL
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

            score = self.scores[idx][-1] * alpha

            # angle = abs(self.cue_shot_vec.get_angle_degrees_between(
                # self.draw_stuff["hit_points_best"] - cb))

            #herherher
            # self.rewardfunc = ['angle_hitpoint', score]
            # print(self.scores)

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
        # print(x, y)
        # ax[0] = plt.scatter(x, y)

        if self.winning_condition_met():
            self.win_list.append('r')
        else:
            self.win_list.append('b')

        t = [i[0] for i in self.actions]
        v = [i[1] for i in self.actions]
        # print(t)
        # print(v)
        # color = 'r' if self.winning_condition_met() else 'b'
        plt.scatter(t, v, c=self.win_list)
        # ax[1] = plt.plot(range(self.steps_taken + 1), v)

        plt.show()

    def out_of_bounds(self):
        obs = self._get_obs()

        for i in range(0, len(obs[::3]), 3):
            if (obs[i] + 1e-2 < 0) or (obs[i] - 1e-2 > 1) or (obs[i + 1] + 1e-2 < 0) or (obs[i + 1] - 1e-2 > 1):
                return True

        return False

    def best_shot_criteria(self, best_vectors):

        # vectors, ball_number = best_vectors
        # vectors, pocket_vectors, ball_numbers = zip(*best_vectors)
        hit_vectors, pocket_id, ball_number = zip(*best_vectors)

        # print(ball_number)

        # Hit vector that maximizes angle window from ball to pocket
        scores = []

        # print(pocket_id)
        # print([self.pocket_ids[pocket_id[i]] for i in range(len(best_vectors))])
        self.scores = []
        # print(self.random_state)
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

            # print(ball_number[i],self.pocket_ids[pocket_id[i]],b_id,p_id,multiplier)

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
            # print(i, round(res,2),round(cos_weight,2),multiplier)

        best = np.argmax(scores)
        bestscore = np.max(scores)
        # print(f"Best window: {window[best]}")
        # best = np.argmin([vec.length for vec in vectors])
        # best = np.argmin([abs(vec.get_angle_degrees_between(poc_vec)) for vec, poc_vec in zip(vectors, pocket_vectors)])

        # best = np.random.randint(0, len(best_vectors))
        # return self.normalize_vector(best_vectors[best]), best
        # print(vectors[best], best)
        return hit_vectors[best], best, bestscore

    def is_straight_line(self, main_pos, target_pos, exclude=[],include=[]):
        # Main_pos: Vec2d
        # Target_pos: Vec2d
        # exclude: [Vec2d, Vec2d]  -- list of points to exclude other than main and target

        main_pos = Vec2d(*main_pos)
        target_pos = Vec2d(*target_pos)

        def point_on_line_seg(a, b, p):
            # a is start of line segment
            # b is end of line segment
            # p in point we want to find the distance to
            # print(p)
            a = Vec2d(*a)
            b = Vec2d(*b)
            p = Vec2d(*p)
            ap = p - a
            ab = b - a
            # print(ap, ab)
            # t = np.dot(ap, ab) / np.dot(ab, ab)
            t = ap.dot(ab) / ab.get_length_sqrd()
            # if you need the the closest point belonging to the segment
            t = max(0, min(1, t))
            point = a + (t * ab)
            dist = p.get_distance(point)
            return dist
        
    
        def hits(item, exlist,s,r=2):
            # if s=="gho": print(len(item))
            try: 
                [ball.body.position for ball in item]
            except:
                pos = np.array([Vec2d(*position) for position in item
                                      if Vec2d(*position) not in exlist]).reshape(-1, 2)
            else:
                pos = np.array([ball.body.position for ball in item 
                                      if ball.body.position not in exlist]).reshape(-1, 2)
            
            dists = [point_on_line_seg(exlist[0], exlist[1], ball) for ball in pos]
            hit = sum([abs(d) <= (r * BALL_RADIUS) for d in dists]) > 0
            
            # print(s,hit,end="")
            
            # if not hit and s=="gho":
            #     for i in range(len(pos)):
            #         print(dists[i] , pos[i],end="")
            
            # print()
            return hit
        
    # Find distance from other balls to line
        # ball_pos = np.array([ball.body.position for ball in self.balls
        #                      if ball.body.position not in [main_pos, target_pos] + exclude]).reshape(-1, 2)
        # dists_ball = [point_on_line_seg(main_pos, target_pos, ball) for ball in ball_pos]
        # hitsball = sum([abs(d) <= 2 * BALL_RADIUS for d in dists_ball]) > 0
        
        

        # ghost_pos = np.array([position for position in self.ghost_balls
        #                      if position not in [main_pos, target_pos] + exclude]).reshape(-1, 2)
        # dists_ghosts = [point_on_line_seg(main_pos, target_pos, ball) for ball in ghost_pos]
        # hitsghost = sum([abs(d) <= 2 * BALL_RADIUS for d in dists_ghosts]) > 0
        
        # print(dists_ghosts)

        # Pocket dist
        # pock_pos = np.array([Vec2d(*target_point) for target_point in self.target_points
        #                      if Vec2d(*target_point) not in [main_pos, target_pos] + exclude]).reshape(-1, 2)

        # # dists_pocket = np.cross(target_pos - main_pos, pock_pos - main_pos) / np.linalg.norm(target_pos - main_pos)
        # dists_pocket = [point_on_line_seg(main_pos, target_pos, pock) for pock in pock_pos]

        # Find distance from cushions to line
        # cushion_pos = np.array([Vec2d(*x) for x in self.cushion_corners.reshape(-1, 2)
        #                         if Vec2d(*x) not in [main_pos, target_pos] + exclude]).reshape(-1, 2)

        # # dists_cushions = np.cross(target_pos - main_pos, cushion_pos - main_pos) / np.linalg.norm(target_pos - main_pos)
        # dists_cushions = [point_on_line_seg(main_pos, target_pos, cush) for cush in cushion_pos]

        # Check if the line is obstructed
        # dists = dists_ball + dists_ghosts + dists_pocket + dists_cushions

        # if sum([abs(d) <= 2 * BALL_RADIUS for d in dists]) > 0:
            
        exlist = [main_pos, target_pos] + exclude
        
        
        # hitsball = hits(self.balls, exlist)
        # hitsghost = hits(self.ghost_balls, exlist)
        # hitspocket = hits(self.target_points, exlist)
        # hitscush = hits(self.cushion_corners.reshape(-1, 2), exlist)
        
        
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
        
        # elif len(include) != 0:
        #     for p in include:
        #         for i, line in enumerate(CUSHION_INNER_LINES):
        #             dist = p[int(i>=2)] - line
        #             if abs(dist) < 2*BALL_RADIUS:
        #                 if i < 2:   fakehitpoint = (line-dist,p[1])
        #                 else:       fakehitpoint = (p[0],line-dist)
                    
        #                 if hits([fakehitpoint], exlist,"fake",r=1):
        #                     self.trash_lines.append([main_pos,target_pos,"fake"])
        #                     self.fakehits.append([p,fakehitpoint])
        #                     print("der var fake lol")
                    

                
            
        # elif hits(self.cushion_corners.reshape(-1, 2), exlist):
        #     self.trash_lines.append([main_pos,target_pos,"cushion"])
        #     return False
            
        # if hitsball or hitsghost or hitspocket or hitscush:
        #     self.trash_lines.append([main_pos,target_pos])
        #     return False  # Line is obstructed
        return True  # The line has no obstructions
        
    
    def add_ghosts(self):
        self.ghost_balls = []
        self.ghost_opponents = []
        
        for ball in self.balls:
            # print(ball.number)
            # if ball.ballclass != self.suit:  # Only consider your own suit
            #     continue

            real_ball_pos = ball.body.position

            ghosts = []
            ghost_opponents = []
            if self.bank_shots:
                for i, line in enumerate(CUSHION_INNER_LINES):
                    dist = real_ball_pos[int(i>=2)] - line
                    if dist > BALL_RADIUS or dist < -BALL_RADIUS:
                        if i < 2:   coord = (line-dist,real_ball_pos[1])
                        else:       coord = (real_ball_pos[0],line-dist)

                        if ball.ballclass == self.suit: ghosts.append(coord)
                        elif ball.ballclass != self.cue_ball.ballclass: ghost_opponents.append(coord)

            self.ghost_balls = self.ghost_balls + ghosts
            self.ghost_opponents = self.ghost_opponents + ghost_opponents
        
    
    def find_best_shot(self):
        # Find lines from balls to pockets
        # lines_to_pockets = [[False] * 6 for _ in range(len(self.balls))]
        # cue_to_ball = [False * (len(self.balls) - 1)]

        # vector_and_pos = [[] for _ in range(self.num_balls - 1)]
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
            
            # print(ball.number)
            # if ball.ballclass != self.suit:  # Only consider your own suit
            #     continue

            # real_ball_pos = ball.body.position

            # ghosts = []
            # ghost_opponents = []
            # if self.bank_shots:
            #     for i, line in enumerate(CUSHION_INNER_LINES):
            #         dist = real_ball_pos[int(i>=2)] - line
            #         if dist > BALL_RADIUS or dist < -BALL_RADIUS:
            #             if i < 2:   coord = (line-dist,real_ball_pos[1])
            #             else:       coord = (real_ball_pos[0],line-dist)

            #             if ball.ballclass == self.suit: ghosts.append(coord)
            #             elif ball.ballclass != self.cue_ball.ballclass: ghost_opponents.append(coord)

            # self.ghost_balls = self.ghost_balls + ghosts
            # self.ghost_opponents = self.ghost_opponents + ghost_opponents

            if ball.ballclass != self.suit:  # Only consider your own suit from here
                continue
            
            for ghostnum, ball_pos in enumerate([real_ball_pos] + self.ghost_balls):
                # Find good pockets
                for i, pocket in enumerate(self.target_points):
                    pocket_pos = Vec2d(*pocket)

                    # lines_to_pockets[ball.number][i] = True

                    # Calculate pos the cue should hit
                    pocket_vec = (pocket_pos - ball_pos).normalized()
                    hit_pos = ball_pos - ((2 - 0) * BALL_RADIUS * pocket_vec)
                    # hit_pos = hit_pos.int_tuple

                    cue2hit_vector = (hit_pos - cue_pos).normalized()
                    theta = cue2hit_vector.get_angle_degrees_between(pocket_vec)

                    self.hit_points.append([Vec2d(*hit_pos), theta])  # Feasible hit_points
                    all_vectors.append([hit_pos - cue_pos, i, ghostnum*100 + ball.number])
                    if (theta > THETA_LIMIT) or (theta < -THETA_LIMIT):  # Bad pocket
                        continue

                    # if (hit_pos[0] <= LOWER_X) or (hit_pos[0] >= UPPER_X) or (hit_pos[1] <= LOWER_Y) or (hit_pos[1] >= UPPER_Y):
                        # continue

                    # print(f"theta: {theta}")
                    # Check if there exists a line from ball to pocket
                    if self.is_straight_line(ball_pos, pocket_pos,include=[hit_pos]):

                        # Check if there exists a line from cue to hit
                        if self.is_straight_line(cue_pos, hit_pos, exclude=[ball_pos]):

                            self.good_hit_points.append(Vec2d(*hit_pos))
                            self.draw_stuff["hit_points"].append(hit_pos)
                            self.draw_stuff["hit_points_details"].append([ball_pos,pocket_pos])
                            # self.good_vectors.append((hit_pos - cue_pos, pocket_vec, ball.number))
                            self.good_vectors.append([hit_pos - cue_pos, i, ghostnum*100 + ball.number])

        # ball for loop end
        # print(f"Good vectors: {[self.normalize_vector(x) for x in self.good_vectors]}\n")
        if len(self.good_vectors) != 0:
            self.best_shot, best, self.best_score = self.best_shot_criteria(self.good_vectors)
            self.draw_stuff["hit_points_best"] = self.draw_stuff["hit_points"][best]
            self.draw_stuff["draw_hit_points"] = True
            # bh = self.draw_stuff["hit_points_best"]
            # print(f"Best hitpoint: {bh}")

            # target_number = self.good_vectors[best][2]
            # self.target_ball = self.balls[target_number]
            # print(self.best_shot)
        else:
            # hit_vectors, thetas = zip(*self.hit_points)
            # self.best_shot = hit_vectors[np.argmin(thetas)] - cue_pos
            
            self.best_shot, best, self.best_score = self.best_shot_criteria(all_vectors)
            # self.draw_stuff["hit_points_best"] = self.draw_stuff["hit_points"][best]
            # self.draw_stuff["draw_hit_points"] = True

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
            ball_vector = 2 * BALL_RADIUS * ball_vector.normalized()
            w1 = (ball_pos + ball_vector.rotated_degrees(90 + 5) - cue_pos).angle_degrees
            w2 = (ball_pos + ball_vector.rotated_degrees(-90 - 5) - cue_pos).angle_degrees

            if (min(w1, w2) <= self.angle) and (self.angle <= max(w1, w2)) and self.is_straight_line(cue_pos, ball_pos):
                self.cue_alpha = 0
            else:
                # self.cue_alpha = min([abs(self.angle - w1), abs(self.angle - w2), self.cue_alpha])
                self.cue_alpha = min([abs(self.angle - (w1+w1)/2), self.cue_alpha])
            # self.window_cue_angles[ball.number] = [w1.angle_degrees, w2.angle_degrees]

            # if ball_tracking["init_velocity"][ball.number] != Vec2d.zero():
            #     # Find window from ball to pockets

        # Find window from target ball to every pocket
        self.target_alpha = np.inf
        if self.tracking["target_ball"] is not None:
            target_ball = self.tracking["target_ball"]
            target_pos = self.ball_tracking["start_positions"][target_ball.number]
            target_angle = self.ball_tracking["init_velocity"][target_ball.number].angle_degrees
        else:
            return None

        for corners in CUSHION_CORNERS:
            corner1, corner2 = corners

            pocket_target = Vec2d(*np.array([corner1, corner2]).mean(axis=0))
            # Find vector from pocket_midpoint to corner
            r1 = Vec2d(*(corner1 - pocket_target)).normalized()
            r2 = Vec2d(*(corner2 - pocket_target)).normalized()

            # Find points that lie on the line between corners and is 1 ball radius away from actual corner
            r1 = corner1 - BALL_RADIUS * r1
            r2 = corner2 - BALL_RADIUS * r2

            w1 = Vec2d(*(r1 - target_pos)).angle_degrees
            w2 = Vec2d(*(r2 - target_pos)).angle_degrees

            # print(f"target_pos: {target_pos}")
            # print(f"pocket_target: {pocket_target}")
            if (min(w1, w2) <= target_angle) and (target_angle <= max(w1, w2)) and self.is_straight_line(target_pos, pocket_target):
                self.target_alpha = 0
            else:
                # self.target_alpha = min([abs(target_angle - w1), abs(target_angle - w2), self.target_alpha])
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
            valid_shot = int(np.round((hit_vec.angle_degrees + 180)*ANGLE_PRECISION))
            
            mask[valid_shot] = True
        
        # best_shot = int(np.round((self.best_shot.angle_degrees + 180)*ANGLE_PRECISION))
        # mask[best_shot] = True
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
            "ball_pocketed": [],  # 1: solids, 2: stripes, 3: cue, 4: black
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
                # print(x.pivot)
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

        self.add_table()
        self.add_balls()
        # Implement note: change black to your suit when all your balls are pocketed
        if sum([ball.ballclass == self.suit for ball in self.balls]) == 0:
            for ball in self.balls:
                if ball.ballclass == 4:
                    ball.ballclass = self.suit
        self.lost = False
        self.win = False
        self.draw_stuff["draw_hit_points"] = False
        self.window_vectors = []
        self.prev_pocketed_balls = []
        

        # self.actions = [[0, 0]]
        # ======== SETUP TRACKING ======== #
        # print(f"Current suit: {self.suit}")
        self.reset_tracking()

        self.find_best_shot()
        # self.number_hit_point = len(self.good_hit_points)

        # ### TEST ###
        # center_alpha = (self.balls[0].body.position - self.cue_ball.body.position).angle_degrees
        # self.test_alphas = np.linspace(center_alpha-3, center_alpha+3, num=60)
        # self.number_hit_point = len(self.test_alphas)
        # self.current_hit_point = 0
        # if not self.replay_state:
        #     self.current_hit_point = 0
        #     self.hit_success.append([False]*self.number_hit_point)

        self.drawing_state = "start"
        if self.obs_type == "image" or not self.training:
            self.render()

        observation = self._get_obs()

        info = self._get_info()
        # if self.obs_type == "image":
            # plt.imshow(observation.reshape(self.height, self.width, 3))
            # plt.show()
        return observation, info

    def step(self, action):
        
        # plt.show()
        # print("\nnewstep\n")
        # print("balls",[ball.body.position for ball in self.balls])
        # print("ghost",self.ghost_balls)
        # print(self.target_points)

        self.window_vectors = []
        self.fakehits = []

        self.find_best_shot()
        # print(f"Number of good hit_points: {len(self.good_vectors)}")
        self.n_hp.append(len(self.good_hit_points))
        
        self.number_hit_point = len(self.good_hit_points)
        if not self.replay_state:
            self.current_hit_point = 0
            self.hit_success.append([False]*self.number_hit_point)

        self.actions.append(action)
        self.angle, self.force = action


        if self.algorithm == "TD3":
            if self.cheat_force: self.force = 1

            self.angle *= N_ANGLES / (2 * ANGLE_PRECISION)  # angle maps from [-1, 1] -> [-180, 180]
            self.force = (175 * self.force + 5) * ZOOM  # Force maps from [0, 1] -> [5, 180] * ZOOM

        else:
            if self.cheat_force: self.force = FORCE-1

            self.angle = self.angle / ANGLE_PRECISION - 180
            self.force = (245/(FORCE-1) * self.force + 5) * ZOOM # Force will be mapped from [0, FORCE-1] -> [5, 180] * ZOOM
            # 175, 245
        # self.angle = np.degrees(np.arccos(np.dot(self.best_shot, np.array([1, 0]))))
        # self.angle = np.degrees(self.return_angle(self.best_shot))
        # self.angle = 1 / 3 * round(self.best_shot.angle_degrees / (1 / 3)) % 360

        if self.cheat_angle:
            self.angle = self.best_shot.angle_degrees

        if self.sigma != 0:
            self.angle += np.random.normal(loc=0, scale=self.sigma)

        # Test all hitpoints
        # if (self.number_hit_point > 0) and self.test_all_hp:
        #     if self.current_hit_point < (self.number_hit_point - 1):
        #         self.angle = (self.good_hit_points[self.current_hit_point] - self.cue_ball.body.position).angle_degrees
        #         # self.angle = self.test_alphas[self.current_hit_point]
        #         # self.current_hit_point += 1
        #         self.replay_state = True
        #     else:
        #         # self.current_hit_point += 1  # For the caption
        #         self.angle = (self.good_hit_points[self.number_hit_point-1] - self.cue_ball.body.position).angle_degrees
        #         # self.angle = self.test_alphas[self.current_hit_point]
        #         self.replay_state = False

        self.reward = 0
        # self.angle = 169.74
        # self.force = 111.37910344
        ## reset first_cue_contact
        self.ball_collision_handler.data["tracking"]["first_cue_contact"] = None
        self.pocket_collision_handler.data["tracking"]["first_cue_contact"] = None
        self.cushion_collision_handler.data["tracking"]["first_cue_contact"] = None
        # self.tracking["first_cue_ball"] = None
        # self.force = 50
        # self.angle = [45, 45 + 90][np.random.randint(2, size=1)[0]]

        # self.angle, self.force = 180, 1500
        # self.steps_taken += 1
        # self.total_steps += 1
        # self.steps_pr_1000 += 1
        x_impulse = np.cos(np.radians(self.angle))
        y_impulse = np.sin(np.radians(self.angle))
        self.cue_shot_vec = Vec2d(self.force * x_impulse, self.force * y_impulse)

        self.cue_ball.body.activate()
        pymunk.Body.update_velocity(self.cue_ball.body, self.cue_shot_vec, damping=0, dt=1)

        self.ball_tracking["init_velocity"][self.cue_ball.number] = self.cue_ball.body.velocity
        self.ball_tracking["calc_velocity"][self.cue_ball.number] = self.cue_ball.body.velocity

        self.terminated = True
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
            # self.ball_tracking["cue_pos"].append(self.cue_ball.body.position)
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

        # print("\nShot loop ended")
        # print(f"Steps taken: \t{self.steps_taken}")
        # If won
        # if self.winning_condition_met():
        #     # print('[w]',end=" ")
        #     self.wins += 1
        #     self.win_pr_1000 += 1
        #     self.terminated = True
        #     self.reward += 1
        #     # self.total_reward += 1

        # elif self.steps_taken >= NUM_SHOTS:
        #     self.loses += 1
        #     self.terminated = True

        # self.find_windows()
        self.no_suits_left = (sum([ball.ballclass == self.suit for ball in self.balls]) == 0)
        if self.num_balls >= 2:
            if self.no_suits_left and (not self.tracking["cue_ball_pocketed"]):
                # If all your balls have been pocketed (+ black, since it changes to your suit in the end), you win
                self.win = True
                self.terminated = True
                self.tracking["is_success"] = True
    
            elif (len(self.tracking["ball_pocketed"]) != 0):
                pocketed_classes = [ball.ballclass for ball in self.tracking["ball_pocketed"]]
                # Sort classes by importance (black > white > your suit > other)
                pocketed_classes.sort(reverse=True)
    
                if pocketed_classes[0] == 4:
                    # Black was pocketed -> you loose the game
                    self.terminated = True
                    self.lost = True
                elif pocketed_classes[0] == 3:
                    # Cue was pocketed -> you miss your turn
                    self.terminated = True
                    if self.no_suits_left:
                        # If no suits left and cue ball pocketed -> you loose
                        self.lost = True
    
                elif pocketed_classes[0] == self.suit:
                    # Your suit was pocketed -> you gain another shot
                    self.terminated = False
    
            else:
                # Enemy suit was pocketed or none was -> you miss your turn
                self.terminated = True
        
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

        # if self.obs_type == "image":
        #     cv2.imshow("im", self.observation)
            # plt.show()

        self._get_reward()
        self.prev_reward = self.reward
        # print(self.reward)
        self.stop_time = time.perf_counter()

        if not self.replay_state:
            self.steps_taken += 1
            # self.total_steps += 1

        if (len(self.good_hit_points) == 0) and (not self.win) and (not self.tracking["cue_ball_pocketed"]):
            # print(f"state: {self.random_state}")
            # print(f"good_hit_points: {len(self.good_hit_points)}")
            # print(f"win: {self.win}")
            # print(self.tracking["cue_ball_pocketed"])
            # print()
            # self.failed_states.append(self.random_state)
            self.no_hitpoint_counter += 1

        self.cue_pocketed += self.tracking["cue_ball_pocketed"] and self.no_suits_left

        # Test all hp
        if self.test_all_hp:
            if (len(self.hit_success[self.random_state]) != 0):
                self.hit_success[self.random_state][self.current_hit_point] = self.no_suits_left

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
            if self.test_all_hp: self.close()
            # self.close()
            # print(f"Wins: \t{self.wins}\t {self.wins/self.total_steps*100}%")

        if self.truncated:
            print(f"Truncation has occured at state: {self.random_state}")
            # self.pretty_print_dict(self.get_attrs())

        # if self.reward > 900:
        #     print(self.get_attrs())
        #     self.close()

        # # Shift ball numbers down
        for ball in self.tracking["ball_pocketed"]:
            number = ball.number
            for b in self.balls:
                if b.number >= number:
                    b.number = b.number - 1
        
        # if (self.terminated and ((not self.win) or self.lost) ) and not self.Continue and not self.training: #and (self.current_hit_point < 1)
        #     self.replay_state = True


        # self.current_hit_point += 1
        
        for ball in self.tracking["ball_pocketed"]:
            self.prev_pocketed_balls.append(ball.ballclass)
        
        self.tracking["blue_in"] = self.no_suits_left
        self.tracking["blue_and_cue_in"] = self.no_suits_left and self.tracking["cue_ball_pocketed"]
        info = self._get_info()
        
        self.reset_tracking()
        if self.terminated or self.truncated:
            self.total_steps += 1
        # print(self.terminated or self.truncated)
        # print(self.total_steps)
        # print(self.tracking["first_cue_contact"])
        # print(f"Terminated: \t{self.terminated}")
        # print(f"Truncated:  \t{self.truncated}")
        # print(f"observation: \t{observation}")
        # self.plot_parms()

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
                color = BESTSHOTCOLOR
            else:
                color = SHOTCOLOR

            # self.screen.fill(RED, (pos, (5*ZOOM,5*ZOOM)))

            # vec = self.ball_tracking["start_positions"][self.target_ball.number] - pos
            # start = pos
            # end = pos + vec*100*ZOOM
            start = self.ball_tracking["start_positions"][self.cue_ball.number]
            end = pos
            if pos == self.draw_stuff["hit_points_best"]:
                bestline = [start,
                            end,
                            self.draw_stuff["hit_points_details"][num][0],
                            self.draw_stuff["hit_points_details"][num][1]]

            if self.view_lines:
                pygame.draw.circle(self.screen, color, pos, 4*ZOOM)
                pygame.draw.line(self.screen, color,
                                  start, end, width=int(2 * ZOOM))
                pygame.draw.line(self.screen, color,
                                  self.draw_stuff["hit_points_details"][num][0],
                                  self.draw_stuff["hit_points_details"][num][1], width=int(2 * ZOOM))

        if self.view_lines:
            font = pygame.font.SysFont('didot.ttc', 20)

            self.screen.blit(font.render("Best Shot", True, BESTSHOTCOLOR), (100,100))
            self.screen.blit(font.render("Possible Shots", True, SHOTCOLOR), (100,120))
            self.screen.blit(font.render("Interupted Shots", True, INTERUPTEDCOLOR), (100,140))

        # Trash lines
            for i,line in enumerate(self.trash_lines):
                if line[2] == "ball":
                    trashcol = RED
                elif line[2] == "ghost":
                    trashcol = GRAY
                elif line[2] == "ghost_opp":
                    trashcol = MAGENTA
                elif line[2] == "pocket":
                    trashcol = CYAN
                elif line[2] == "cushion":
                    trashcol = PURPLE
                pygame.draw.line(self.screen, trashcol,
                                  line[0], line[1], width=int(2 * ZOOM))

                # text = font.render(str(i), True, INTERUPTEDCOLOR)
                # self.screen.blit(text, line[1])

        # Target points
        # for count,pos in enumerate(self.target_points):
        #     pygame.draw.circle(self.screen, CYAN, pos, 4*ZOOM)

            # font = pygame.font.SysFont('didot.ttc', 20)
            # text = font.render(str(self.pocket_ids[count]), True, RED)
            # self.screen.blit(text, (pos[0]+2*BALL_RADIUS,pos[1]+5*BALL_RADIUS))

        # Cushion Corners
        # for count,positions in enumerate(self.cushion_corners):
        #     for pos in positions:
        #         pygame.draw.circle(self.screen, YELLOW, pos, 4*ZOOM)

                # font = pygame.font.SysFont('didot.ttc', 20)
                # text = font.render(str(count), True, BLACK)
                # self.screen.blit(text, (pos[0]+2*BALL_RADIUS,pos[1]+5*BALL_RADIUS))

        # Draw windows


        txts = [[0] for _ in self.target_points]

        # print(self.pocket_value_counter)
        if VIEW_MIRRORS:
            for i,vecset in enumerate(self.window_vectors):
                # ball, vec1, vec2, windowsize
                # pygame.draw.line(self.screen, BLACK, vecset[0], vecset[1], width=int(3 * ZOOM))
                # pygame.draw.line(self.screen, BLACK, vecset[0], vecset[2], width=int(3 * ZOOM))

                font = pygame.font.SysFont('didot.ttc', 28)

                # Pocket ids
                # text = font.render("{:.2f}".format(vecset[3]), True, RED)
                # self.screen.blit(text, vecset[1])
                # prod_scores = np.prod(self.scores[i][:3])
                if len(self.scores) == len(self.window_vectors):
                    prod_scores = self.scores[i][-1]

                    if prod_scores == self.best_score:
                        text = font.render("{0:.2f}".format(prod_scores), True, BESTSHOTCOLOR)
                    else:
                        text = font.render("{0:.2f}".format(prod_scores), True, BLACK)
                    txts[vecset[4]].append([text,vecset[1]])

        for pock in txts:
            for i,txt in enumerate(pock):
                if txt!=0: self.screen.blit(txt[0], (txt[1][0]+20,txt[1][1]+(i*20)))

        if self.replay_state:
            font = pygame.font.SysFont('didot.ttc', 40)
            text = font.render("Replay", True, RED)
            self.screen.blit(text, (LOWER_X,20))
        if self.Continue:
            font = pygame.font.SysFont('didot.ttc', 40)
            text = font.render("Continue", True, RED)
            self.screen.blit(text, (LOWER_X,20))


        if self.drawing_state == "end":
            # Draw cue shot
            vec = self.ball_tracking["init_velocity"][self.cue_ball.number]
            start = self.ball_tracking["start_positions"][self.cue_ball.number]
            end = start + vec * ZOOM * 2
            # pygame.draw.line(self.screen, CYAN,
            #                   start, end, width=int(1 * ZOOM))

            
        if self.bank_shots:
            for i in range(4):
                if i < 2: pygame.draw.line(self.screen, TABLE_SIDE_COLOR, (CUSHION_INNER_LINES[i],0), (CUSHION_INNER_LINES[i],FULL_SCREEN_HEIGHT))
                else: pygame.draw.line(self.screen, TABLE_SIDE_COLOR, (0,CUSHION_INNER_LINES[i]), (FULL_SCREEN_WIDTH,CUSHION_INNER_LINES[i]))

            if VIEW_MIRRORS:
                for pos in self.ghost_balls:
                    pygame.draw.circle(self.screen, YELLOW, pos, BALL_RADIUS)

                for pos in self.ghost_opponents:
                    pygame.draw.circle(self.screen, MAGENTA, pos, BALL_RADIUS)


        # Draw collision points
        for cp_pair, dist in zip(self.tracking["contact_point"], self.tracking["contact_point_dist"]):
            # print(cp_pair)
            for i, pos in enumerate(cp_pair):
                # self.screen.fill(BLUE, (pos, (5 * ZOOM, 5 * ZOOM)))
                # pygame.draw.circle(self.screen, (255, 153, 255), pos, 3)

                if i == 0:
                    font = pygame.font.SysFont('didot.ttc', 20)
                    text = font.render(str(round(dist, 2)), True, RED)
                    # self.screen.blit(text, (pos[0]+3*BALL_RADIUS,pos[1]+1*BALL_RADIUS))

        # Track balls position
        # print(self.cue_ball.body.position.int_tuple)
        for i in range(len(self.ball_tracking["ball_pos"])):

            for pos in self.ball_tracking["ball_pos"][i]:
                if self.ball_tracking["ball_classes"][i] == self.suit:
                    pygame.draw.circle(self.screen, BLUE, pos, 3)
                elif self.ball_tracking["ball_classes"][i] == 1:
                    pygame.draw.circle(self.screen, SOLID_COLOR, pos, 3)
                elif self.ball_tracking["ball_classes"][i] == 3:
                    pygame.draw.circle(self.screen, WHITE, pos, 3)

        # Draw normal vectors
        # for i, vec in enumerate(self.tracking["normal_vectors"]):
        #     # print(self.tracking["normal_vectors"])
        #     # print(i)
        #     # print(vec)
        #     # print(self.tracking["contact_point"])
        #     # print()
        #     start = Vec2d(*np.mean(self.tracking["contact_point"][i],axis=0))
        #     end = start -vec * ZOOM * 200
        #     pygame.draw.line(self.screen, (255, 153, 255),
        #                      start, end, width=int(3 * ZOOM))

        for i in range(len(self.ball_tracking["start_positions"])):
            start = self.ball_tracking["start_positions"][i]  # Vec2d
            end = start + self.ball_tracking["init_velocity"][i] * 2  # Vec2d
            end2 = start + self.ball_tracking["calc_velocity"][i] # Vec2d
            ball_class = self.ball_tracking["ball_classes"][i]

            # pygame.draw.line(self.screen, DRAW_SHOT_COLOR[ball_class - 1],
            #                   start, end, width=int(3))

            # if i != self.cue_ball.number:
            #     pygame.draw.line(self.screen, BLACK,
            #                       start, end2, width=int(2))

            # self.screen.fill(SUIT_COLORS[ball_class - 1], (start, (10*ZOOM,10*ZOOM)))
            pygame.draw.circle(self.screen, SUIT_COLORS[ball_class - 1], start, BALL_RADIUS*1.3, width=int(1*ZOOM))

        
        # draw fake hitpoints
        for h in self.fakehits:
            pygame.draw.line(self.screen, CYAN, h[0],h[1], width=int(1*ZOOM))





        #draw best line on top
        # bestline = None
        if bestline is not None:
            # pygame.draw.line(self.screen, DRAW_SHOT_COLOR[ball_class - 1],
            #                   bestline[0], bestline[1], width=int(3))
            # pygame.draw.line(self.screen, DRAW_SHOT_COLOR[ball_class - 1],
            #                   bestline[2], bestline[3], width=int(3))


            path1 = (bestline[0],bestline[1])
            path2 = (bestline[2],bestline[3])

            if self.bank_shots:
                for i, line in enumerate(CUSHION_INNER_LINES):
                    y = int(i>1)
                    
                    for path in [path1,path2]:
                        dist1 = path[0][y] - line   # afstand fra x1 til line
                        dist2 = path[1][y] - line   # afstand fra x2 til line
                        
                        show = sum([[c[y]<line, c[y]>line, c[y]>line, c[y]<line][i] for c in path])!=0
                        
                        # if show:
                        #     pygame.draw.line(self.screen, BESTSHOTCOLOR,
                        #                           (path[0][0] - (not y) * 2*dist1, 
                        #                             path[0][1] - y * 2*dist1),
                                                  
                        #                           (path[1][0] - (not y) * 2*dist2, 
                        #                             path[1][1] - y * 2*dist2),
                                                  
                        #                           width=int(3))
        
        # pocketed_now = self.prev_pocketed_balls
        # print(pocketed_now)
        
        # for ball in self.tracking["ball_pocketed"]:
        #     pocketed_now.append(ball.ballclass)
        
        # display pocketed balls
        blues = 0
        for i,ball in enumerate(self.prev_pocketed_balls + [ball.ballclass for ball in self.tracking["ball_pocketed"]]):
            blues += int(ball==2)
            if ball==2 and blues > self.numblues: pygame.draw.circle(self.screen, BLACK, (LOWER_X+(3*BALL_RADIUS)*(i+1),UPPER_Y+(5*BALL_RADIUS)), BALL_RADIUS)
            else: pygame.draw.circle(self.screen, SUIT_COLOR[ball], (LOWER_X+(3*BALL_RADIUS)*(i+1),UPPER_Y+(5*BALL_RADIUS)), BALL_RADIUS)
                

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
                self.screen.fill(BLUE, (pos, (5 * ZOOM, 5 * ZOOM)))
                pygame.draw.circle(self.screen, YELLOW, pos, 2*ZOOM)

        # Draw circle around balls + initial velocity
        for i in range(len(self.ball_tracking["start_positions"])):
            start = self.ball_tracking["start_positions"][i]  # Vec2d
            end = start + self.ball_tracking["init_velocity"][i] * 2  # Vec2d
            end2 = start + self.ball_tracking["calc_velocity"][i] # Vec2d
            ball_class = self.ball_tracking["ball_classes"][i]

            pygame.draw.line(self.screen, DRAW_SHOT_COLOR[ball_class - 1],
                              start, end, width=int(3 * ZOOM))

            if i != self.cue_ball.number:
                pygame.draw.line(self.screen, BLACK,
                                  start, end2, width=int(2 * ZOOM))

            # self.screen.fill(SUIT_COLORS[ball_class - 1], (start, (10*ZOOM,10*ZOOM)))
            pygame.draw.circle(self.screen, SUIT_COLORS[ball_class - 1], start, BALL_RADIUS*1.3, width=int(1*ZOOM))


            # a, b = self.tracking["contact_point"][0:2]
            # start = a
            # end = a + (b - a).normalized() * 100
            # pygame.draw.line(self.screen, BLACK,
            #                  start, end, width=int(3 * ZOOM))

    def redraw_screen(self):
        # if WINDOW_MULTIPLIER != 1: 
        # self.screen.fill(pygame.Color(TABLE_COLOR))
        # else: 
        self.screen.fill(pygame.Color(BG_COLOR))

        pygame.draw.polygon(self.screen, TABLE_COLOR, TABLE_SPACE)

        self.draw_state()
        self.space.debug_draw(self.draw_options)

        sp = " "*5

        captionMAIN = f"Pool {sp} FPS: {self.fps} {sp} Algorithm: {self.algorithm}" + sp
        captionSUIT = f"Your suit: {SUIT_NAMES[self.suit-1]} ({SUIT_COLORS[self.suit-1]})" + sp
        captionREWARD = f"previous reward: {round(self.prev_reward,3)} " + sp
        captionSTATE= f"random_state_nr: {self.random_state}" + sp
        if self.test_all_hp: captionHITPOINT = f"HitPoint: {self.current_hit_point}/{self.number_hit_point}" + sp
        else: captionHITPOINT = ""
        if self.best_score: captionVALUE = f"HitPoint Value: {round(self.best_score,2)}" + sp
        else: captionVALUE = f"HitPoint Value: 0" + sp
        caption = captionMAIN + captionSUIT + captionSTATE + captionREWARD + captionHITPOINT + captionVALUE
        pygame.display.set_caption(caption)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def render(self):
        self.redraw_screen()
        self.process_events()
        # print(self.balls[0].body.velocity)

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
            # "episodes": copy(self.episodes),
            # "episode_reward": copy(self.episode_reward),
            # "episode_steps": copy(self.episode_steps),
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
