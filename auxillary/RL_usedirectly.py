from stable_baselines3 import PPO, TD3, A2C, DDPG, SAC
from pymunk.vec2d import Vec2d
import numpy as np

from auxillary.RL_config_env import *
from auxillary.RL_oracle import Oracle


# from sb3_contrib.ppo_mask import MaskablePPO

def load_RL_no_env(model_path):
    algo = model_path.split('/')[-1].split('.')[0]
    if algo == 'PPO': model = PPO.load(model_path)
    elif algo == 'TD3': model = TD3.load(model_path)
    elif algo == 'DDPG': model = DDPG.load(model_path)
    elif algo == 'SAC': model = SAC.load(model_path)
    elif algo == 'A2C': model = A2C.load(model_path)
    elif algo == 'Oracle': model = Oracle(suit=2)
    # elif algo == 'PPO_masked': model = MaskablePPO.load(model_path)
    return model


def add_ghosts(state):
    ghosts = []
    
    for ball in state:
        real_ball_pos = ball[:2]
        
        for i, line in enumerate([0.0, 1.0, 0.0, 1.0]):  # min_x, max_x, min_y, max_y
            dist = real_ball_pos[int(i>=2)] - line
            if dist > BALL_RADIUS or dist < -BALL_RADIUS:
                if i < 2:   coord = (line-dist,real_ball_pos[1])
                else:       coord = (real_ball_pos[0],line-dist)

                if ball[2] != 3: ghosts.append(coord)  # If ball is not cue ball, add its ghost version
    return ghosts


# TODO: Complete this function
def get_pocket_targets():
    pockets = np.array([[0.0, 0.0],  # left-bot
                        [1.0, 0.0],  # right-bot
                        [1.0, 1.0],  # right-top
                        [0.0, 1.0],  # left-top
                        [0.5, 1.0],  # mid-top
                        [0.5, 0.0]   # mid-bot
                        ])

    extra_targets = []

    for i, line in enumerate([0.0, 1.0, 0.0, 1.0]):  # min_x, max_x, min_y, max_y
        for k,target in enumerate(pockets):
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



def find_best_shot(state, suit=2):
    # Find lines from balls to pockets
    good_vectors = []
    all_vectors = []

    best_shot = None
    hit_points = []
    good_hit_points = []
    
    pockets = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],  # bottom line pockets
                        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]   # upper line pockets
                        ])
    cue_pos = state[state[:, -1] == 3].flatten()[:2]
    assert len(cue_pos) > 0, "No cue ball found."
    
    cue_pos = Vec2d(cue_pos)

    if len(state) == 1:  # There is only the cue ball
        random_int = np.random.randint(0, 6)
        pocket_pos = Vec2d(*pockets[random_int])
        best_shot = pocket_pos - cue_pos
        return np.array([best_shot.angle_degrees, 1.0])  # angle, force
    
    ghost_balls = add_ghosts()

    for ball in state:
        real_ball_pos = ball[:2]

        if ball[2] != suit:  # Only consider your own suit from here
            continue
        
        # TODO: Get target points
        for ghostnum, ball_pos in enumerate([real_ball_pos] + ghost_balls):
            # Find good pockets
            for i, pocket in enumerate(self.target_points):
                pocket_pos = Vec2d(*pocket)

                # Calculate pos the cue should hit
                pocket_vec = (pocket_pos - ball_pos).normalized()
                hit_pos = ball_pos - ((2 - 0) * BALL_RADIUS * pocket_vec)

                cue2hit_vector = (hit_pos - cue_pos).normalized()
                theta = cue2hit_vector.get_angle_degrees_between(pocket_vec)

                self.hit_points.append([Vec2d(*hit_pos), theta])  # Feasible hit_points
                all_vectors.append([hit_pos - cue_pos, i, ghostnum*100 + ball.number])
                if (theta > THETA_LIMIT) or (theta < -THETA_LIMIT):  # Bad pocket
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
        hit = sum([abs(d) <= (r * BALL_RADIUS) for d in dists]) > 0
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
    

# def add_ghosts(self):
#     self.ghost_balls = []
#     self.ghost_opponents = []
    
#     for ball in self.balls:
#         real_ball_pos = ball.body.position

#         ghosts = []
#         ghost_opponents = []
#         if self.bank_shots:
#             for i, line in enumerate(CUSHION_INNER_LINES):
#                 dist = real_ball_pos[int(i>=2)] - line
#                 if dist > BALL_RADIUS or dist < -BALL_RADIUS:
#                     if i < 2:   coord = (line-dist,real_ball_pos[1])
#                     else:       coord = (real_ball_pos[0],line-dist)

#                     if ball.ballclass == self.suit: ghosts.append(coord)
#                     elif ball.ballclass != self.cue_ball.ballclass: ghost_opponents.append(coord)

#         self.ghost_balls = self.ghost_balls + ghosts
#         self.ghost_opponents = self.ghost_opponents + ghost_opponents
    

# def find_best_shot(self):
#     # Find lines from balls to pockets
#     self.good_vectors = []
#     all_vectors = []
#     self.draw_stuff["hit_points"] = []
#     self.draw_stuff["hit_points_details"] = []
#     self.best_shot = None
#     self.hit_points = []
#     self.good_hit_points = []

#     cue_pos = self.cue_ball.body.position

#     if self.num_balls == 1:
#         random_int = np.random.randint(0, 6)
#         pocket_pos = Vec2d(*self.target_points[random_int])
#         self.best_shot = pocket_pos - cue_pos
#         return None
    
#     self.add_ghosts()

#     for ball in self.balls:
#         real_ball_pos = ball.body.position

#         if ball.ballclass != self.suit:  # Only consider your own suit from here
#             continue
        
#         for ghostnum, ball_pos in enumerate([real_ball_pos] + self.ghost_balls):
#             # Find good pockets
#             for i, pocket in enumerate(self.target_points):
#                 pocket_pos = Vec2d(*pocket)

#                 # Calculate pos the cue should hit
#                 pocket_vec = (pocket_pos - ball_pos).normalized()
#                 hit_pos = ball_pos - ((2 - 0) * BALL_RADIUS * pocket_vec)

#                 cue2hit_vector = (hit_pos - cue_pos).normalized()
#                 theta = cue2hit_vector.get_angle_degrees_between(pocket_vec)

#                 self.hit_points.append([Vec2d(*hit_pos), theta])  # Feasible hit_points
#                 all_vectors.append([hit_pos - cue_pos, i, ghostnum*100 + ball.number])
#                 if (theta > THETA_LIMIT) or (theta < -THETA_LIMIT):  # Bad pocket
#                     continue


#                 # Check if there exists a line from ball to pocket
#                 if self.is_straight_line(ball_pos, pocket_pos,include=[hit_pos]):

#                     # Check if there exists a line from cue to hit
#                     if self.is_straight_line(cue_pos, hit_pos, exclude=[ball_pos]):

#                         self.good_hit_points.append(Vec2d(*hit_pos))
#                         self.draw_stuff["hit_points"].append(hit_pos)
#                         self.draw_stuff["hit_points_details"].append([ball_pos,pocket_pos])
#                         self.good_vectors.append([hit_pos - cue_pos, i, ghostnum*100 + ball.number])

#     # ball for loop end
#     if len(self.good_vectors) != 0:
#         self.best_shot, best, self.best_score = self.best_shot_criteria(self.good_vectors)
#         self.draw_stuff["hit_points_best"] = self.draw_stuff["hit_points"][best]
#         self.draw_stuff["draw_hit_points"] = True

#     else:
#         self.best_shot, best, self.best_score = self.best_shot_criteria(all_vectors)
#     return None