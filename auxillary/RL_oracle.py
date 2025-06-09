import numpy as np
from pymunk import Vec2d

import matplotlib.pyplot as plt

class Oracle():
    def __init__(self, suit=2):
        import auxillary.RL_config_env as cfg
        self.cfg = cfg  # RL env config
        self.suit = suit
        
        # Setup pockets (target_points)
        self.pocket_ids = [0,1,2,3,4,5]
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
    
    def remap_to_env_size(self, obs):
    
        obs = obs.reshape(-1, 3)
        
        new_obs = np.array([
            [
            int((self.cfg.UPPER_X - self.cfg.LOWER_X - 2)*x + self.cfg.LOWER_X) + 1,
            int((self.cfg.UPPER_Y - self.cfg.LOWER_Y - 2)*y + self.cfg.LOWER_Y) + 1,
            int(c * 4)
            ] for x,y,c in obs if c != 0
        ])
        
        ball_number = np.arange(1, len(new_obs)+1).reshape(-1, 1)
        return np.hstack((new_obs, ball_number))
    
    def add_ghosts(self, obs):
        self.ghost_balls = []
        self.ghost_opponents = []
        for ball in obs:
            real_ball_pos = ball[:2]
            
            for i, line in enumerate(self.cfg.CUSHION_INNER_LINES):
                dist = real_ball_pos[int(i>=2)] - line
                if dist > self.cfg.BALL_RADIUS or dist < -self.cfg.BALL_RADIUS:
                    if i < 2:   coord = (line-dist,real_ball_pos[1])
                    else:       coord = (real_ball_pos[0],line-dist)

                    if ball[2] == self.suit: self.ghost_balls.append(coord)
                    elif ball[2] != 3: self.ghost_opponents.append(coord)  # If ball is not cue ball, add its ghost version

    
    def best_shot_criteria(self, best_vectors):

        # hit_vectors, pocket_id, ball_number = zip(*best_vectors)
        cue_pos,hit_pos,ball_pos,pocket_pos, pocket_id, ball_number = zip(*best_vectors)
        
        scores = []
        self.scores = []
        ballpos = []

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
                ballpos.append(self.obs[ball_number[i]][:2])
            else:
                ballpos.append(self.ghost_balls[(ball_number[i] // 100) - 1])
            a1 = cus1 - ballpos[i]
            a2 = cus2 - ballpos[i]

            angle_between = abs(a1.get_angle_degrees_between(a2)) / 60

            # window
            # self.window_vectors.append((ballpos,cus1,cus2,angle_between,pocket_id[i]))

            # Cosine_sim weight
            hit_vec = (hit_pos[i] - cue_pos[i]).normalized()
            poc_vec = (Vec2d(*self.target_points[pocket_id[i]]) - ballpos[i]).normalized()

            cos_weight = hit_vec.dot(poc_vec)

            scores.append((angle_between + cos_weight) * multiplier)
            self.scores.append([angle_between,cos_weight,multiplier, scores[-1]])

        best = np.argmax(scores)
        bestscore = np.max(scores)

        self.best_locations = [cue_pos[best],hit_pos[best],ball_pos[best],pocket_pos[best]]

        # self.best_locations = ballpos[best]
        # self.best_pocket_location = Vec2d(*self.target_points[pocket_id[best]])

        return hit_pos[best]-cue_pos[best], best, bestscore

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
                [ball[:2] for ball in item]
            except:
                pos = np.array([Vec2d(*position) for position in item
                                        if Vec2d(*position) not in exlist]).reshape(-1, 2)
            else:
                # print(item)
                # print(exlist)
                pos = np.array([ball[:2] for ball in item 
                                        if Vec2d(*ball[:2]) not in exlist]).reshape(-1, 2)

            dists = [point_on_line_seg(exlist[0], exlist[1], ball) for ball in pos]
            hit = sum([abs(d) <= (r * self.cfg.BALL_RADIUS) for d in dists]) > 0
            return hit

        exlist = [main_pos, target_pos] + exclude        
        
        if hits(self.obs, exlist,"bal"):
            # self.trash_lines.append([main_pos,target_pos,"ball"])
            return False
            
        elif hits(self.ghost_balls, exlist,"gho"):
            # self.trash_lines.append([main_pos,target_pos,"ghost"])
            return False
            
        elif hits(self.ghost_opponents, exlist,"opp"):
            # self.trash_lines.append([main_pos,target_pos,"ghost_opp"])
            return False
            
        elif hits(self.target_points, exlist,"poc"):
            # self.trash_lines.append([main_pos,target_pos,"pocket"])
            return False

        return True  # The line has no obstructions
    
    
    def predict(self, obs, deterministic=True):

        # print('obs before', obs)
        self.obs = self.remap_to_env_size(obs)
        # print('obs after', self.obs)

        # Find lines from balls to pockets
        good_vectors = []
        all_vectors = []
        best_shot = None
        hit_points = []
        good_hit_points = []
        force = 29
        
        cue_pos = self.obs[self.obs[:, -1] == 3].flatten()[:2]
        assert cue_pos.size > 0, "Found no Cue ball. Can't make a prediction"
        cue_pos = Vec2d(*cue_pos)
        # cue_pos = cue_ball.body.position

        if self.obs.shape[0] == 1:
            random_int = np.random.randint(0, 6)
            pocket_pos = Vec2d(*self.target_points[random_int])
            best_shot = pocket_pos - cue_pos
            angle = best_shot.angle_degrees
            
            return np.array([angle, 1]), None  # Return also None to make compatible with sb3 model.predict method

        self.add_ghosts(self.obs)

        for ball in self.obs:
            real_ball_pos = ball[:2]
            ball_number = ball[3]
            if ball[2] != self.suit:  # Only consider your own suit from here
                continue
        
            for ghostnum, ball_pos in enumerate([real_ball_pos] + self.ghost_balls):
                # Find good pockets
                ball_pos = Vec2d(*ball_pos)
                for i, pocket in enumerate(self.target_points):
                    pocket_pos = Vec2d(*pocket)

                    # Calculate pos the cue should hit
                    pocket_vec = (pocket_pos - ball_pos).normalized()
                    hit_pos = ball_pos - ((2 - 0) * self.cfg.BALL_RADIUS * pocket_vec)
                    cue2hit_vector = (hit_pos - cue_pos).normalized()
                    theta = cue2hit_vector.get_angle_degrees_between(pocket_vec)

                    hit_points.append([Vec2d(*hit_pos), theta])  # Feasible hit_points
                    # all_vectors.append([hit_pos - cue_pos, i, ghostnum*100 + ball_number])
                    all_vectors.append([cue_pos, hit_pos,ball_pos,pocket_pos, i, ghostnum*100+ball_number])
                    if (theta > self.cfg.THETA_LIMIT) or (theta < -self.cfg.THETA_LIMIT):  # Bad pocket
                        continue


                    # Check if there exists a line from ball to pocket
                    if self.is_straight_line(ball_pos, pocket_pos, include=[hit_pos]):

                        # Check if there exists a line from cue to hit
                        if self.is_straight_line(cue_pos, hit_pos, exclude=[ball_pos]):

                            good_hit_points.append(Vec2d(*hit_pos))
                            good_vectors.append([cue_pos, hit_pos,ball_pos,pocket_pos, i, ghostnum*100+ball_number])

        # ball for loop end
        if len(good_vectors) != 0:
            best_shot, best, best_score = self.best_shot_criteria(good_vectors)
        else:
            best_shot, best, best_score = self.best_shot_criteria(all_vectors)
        
        self.good_vectores = good_vectors
        angle = best_shot.angle_degrees
        return np.array([angle, 1]), None  # Return also None to make compatible with sb3 model.predict method