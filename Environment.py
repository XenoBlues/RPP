import numpy as np
from spatialmath.base import transl, norm
from CollisionDetection import CapsuleCylinderCollision
from FR5 import FR5
from Utils import traj_interpolate, Cylinder, Capsule, AABB
import roboticstoolbox
from spatialmath import *

rad = np.pi / 180
delta = 0.1  # 距离奖励转折点
omega_t1 = 1  # 距离奖励系数 -100
omega_t2 = -1000  # 距离奖励系数
omega_v = -0.05  # 速度系数 -0.001
omega_acc = -0.05  # 加速度系数
omega_c = 10  # 接近系数
omegas_avoid = np.array([0.1, 0.2, 0.3, 0.4])  # [25, 50, 100, 150]
d_t = 0.01  # 目标距离阈值(米)
d_s = 0.1   # 安全距离(米)
r_gs = [20, 25, 30, 35, 40, 45, 50, 55]
r_g = 50    # 25
r_c = -25   # 20
link_radius = [58, 40, 40, 40]
reward_decay = False


class HREnv:
    def __init__(self, target_points, initial_poses, human_data_path, single_max_step, detect_collision=False,
                 control_mode="segmental", obj_type="human", add_noise=True, loop_human=True):
        self.robot = FR5()
        self.target_points = target_points
        self.initial_poses = initial_poses
        self.target_idx = 0
        self.human_traj_data = np.load(human_data_path)
        self.traj_idx = 0
        self.cur_step = 0
        self.obj_type = obj_type
        self.add_noise = add_noise
        self.min_dis = np.inf
        self.suf_dis = []
        self.target = self.target_points[self.target_idx]
        self.single_step_duration = 0.05
        self.human_time_list = np.linspace(0, 20, 600)
        self.detect_collision = detect_collision
        self.control_mode = control_mode
        self.loop_human = loop_human
        self.single_max_step = single_max_step
        self.global_step = 0
        if self.control_mode == "segmental":
            self.max_step = single_max_step
        self.max_len = []
        self.prev_vel = None
        self.prev_state = None
        self.acc = None
        self.initial_dis = 0
        if self.detect_collision:
            if self.obj_type == "human":
                self.state_Dim = 6 + 6 * 3 + 3 + 10 * 3  # 6 angular vel + 6 joint pos + target pose + 10 human joints pos = 57
        else:
            self.state_Dim = 6 + 6 * 3 + 3  # 6 angular vel + 6 joint pos + target pose = 27

        self.cur_state = np.zeros(self.state_Dim)
        self.action_Dim = 6  # 6 angular vel
        self.robot_links = [Cylinder([0, 0, 0], [1, 1, 1], 135 / 2),
                            Cylinder([0, 0, 0], [1, 1, 1], 100 / 2),
                            Cylinder([0, 0, 0], [1, 1, 1], 80 / 2),
                            Cylinder([0, 0, 0], [1, 1, 1], 80 / 2)]

        self.human_body = [Capsule([0, 0, 0], [1, 1, 1], 250),  # neck spine_naval
                           Capsule([0, 0, 0], [1, 1, 1], 110 / 2),  # elbow_left shoulder_left
                           Capsule([0, 0, 0], [1, 1, 1], 130 / 2),  # wrist_left elbow_left
                           Capsule([0, 0, 0], [1, 1, 1], 130 / 2),  # handtip_left wrist_left
                           Capsule([0, 0, 0], [1, 1, 1], 110 / 2),  # elbow_right shoulder_right
                           Capsule([0, 0, 0], [1, 1, 1], 130 / 2),  # wrist_right elbow_right
                           Capsule([0, 0, 0], [1, 1, 1], 130 / 2)]  # handtip_right wrist_right

        self.obstacle = AABB([-165, -470, 120])
        self.min_dis = np.inf

        self.prev_link_dis = None
        self.prev_target_dis = None

    def update_collider(self, q, p, robot_joint_pos, human_joint_pos):
        x2 = -np.sin(q[0]) * -138
        y2 = np.cos(q[0]) * -138
        x3 = -0.425 * np.cos(q[0]) * np.cos(q[1]) + x2
        y3 = -0.425 * np.sin(q[0]) * np.cos(q[1]) + y2
        p2 = np.array([x2, y2, robot_joint_pos[0, -1]])  # joint2 center
        p3 = np.array([x3, y3, robot_joint_pos[1, -1]])  # joint3 center


        vec0 = (p3 - p2) / np.linalg.norm(p3 - p2)
        tip0 = p3 + vec0 * 58
        base0 = p2 - vec0 * 58
        self.robot_links[0].set(tip0, base0)

        vec1 = (robot_joint_pos[2, :] - robot_joint_pos[1, :]) / np.linalg.norm(
            robot_joint_pos[2, :] - robot_joint_pos[1, :])
        tip1 = robot_joint_pos[2, :] + vec1 * 40
        base1 = robot_joint_pos[1, :] - vec1 * 58
        self.robot_links[1].set(tip1, base1)

        vec2 = (robot_joint_pos[4, :] - robot_joint_pos[3, :]) / np.linalg.norm(
            robot_joint_pos[4, :] - robot_joint_pos[3, :])
        tip2 = robot_joint_pos[3, :] + vec2 * 51
        base2 = robot_joint_pos[3, :] - vec2 * 51
        self.robot_links[2].set(tip2, base2)

        vec3 = (robot_joint_pos[5, :] - robot_joint_pos[4, :]) / np.linalg.norm(
            robot_joint_pos[5, :] - robot_joint_pos[4, :])
        tip3 = robot_joint_pos[4, :] + vec3 * 210
        base3 = robot_joint_pos[4, :] - vec3 * 40
        self.robot_links[3].set(tip3, base3)

        if self.obj_type == "human":
            for j, part in enumerate(self.human_body):
                tip = human_joint_pos[j + 1, :]
                if j == 0:
                    base = human_joint_pos[j, :]
                    up = tip - base
                    up = up / np.linalg.norm(up)
                    part.set(tip + up * 220, base)  # 1 0
                else:
                    base = human_joint_pos[j + 2, :]
                    up = tip - base
                    up = up / np.linalg.norm(up)
                    part.set(tip + up * 20, base - up * 20)  # 2 3

    def collision_detection(self):
        dis_i = []
        b_collision = []
        diffs = []
        if self.obj_type == "human":
            for i, link in enumerate(self.robot_links):
                tmp_dis = []
                tmp_col = []
                tmp_diff = []
                for j, part in enumerate(self.human_body):
                    if isinstance(part, Capsule):
                        dis, diff, b_col = CapsuleCylinderCollision(part, link)
                        tmp_dis.append(dis / 1000.0)
                        tmp_col.append(b_col)
                        tmp_diff.append(diff / 1000.0)
                dis_i.append(min(tmp_dis))
                diffs.append(tmp_diff[np.argmin(tmp_dis)])
                b_collision.append(tmp_col)

        return dis_i, b_collision, diffs

    def reward(self, state):
        """
        :param state: current state
        :return:
        """
        j6_pos = state[21:24]
        target_pos = state[24:27]
        dis_targ = np.linalg.norm(target_pos - j6_pos) / 1000.0  # target distance

        # cal distance between colliders
        if self.detect_collision:
            cur_link_dis, b_collision, diff = self.collision_detection()
            self.min_dis = min(self.min_dis, np.min(cur_link_dis))
            self.suf_dis.append(np.min(cur_link_dis))

        else:
            cur_link_dis = [np.inf, np.inf, np.inf, np.inf]
            b_collision = [False]
            b_avoid = [False]
            diff = []

        r = 0
        suc = False
        fail = False

        if dis_targ <= d_t:  # reach target
            if self.control_mode == "segmental":
                r_g_ = r_g
            r += r_g_
            suc = True

        if True in b_collision:  # collision occur
            r += r_c
            fail = True

        # safety reward
        r_avoid_arr = np.zeros(4)
        d_o = []
        if self.detect_collision:
            for i in range(len(cur_link_dis)):
                if self.obj_type == "human":
                    d = diff[i] - d_s
                else:
                    d = cur_link_dis[i] - (d_s + self.max_len[i])
                if d > 0:
                    d = 0
                d_o.append(d)
                r_avoid_arr[i] = np.log(2 / (1 + np.exp(-2 * d)))
        r_avoid = (r_avoid_arr * omegas_avoid).sum()
        r += r_avoid

        # target distance reward
        r_d = omega_t1 * (1 - np.exp(dis_targ * 2))

        # velocity acceleration reward
        r_v = omega_v * np.linalg.norm(self.cur_state[:6])
        r_acc = omega_acc * np.linalg.norm(self.acc)

        r += (r_d + r_v + r_acc)

        self.prev_link_dis = cur_link_dis
        self.prev_target_dis = dis_targ

        return r, dis_targ, suc, fail

    def reset(self, max_step=None, target_points=None, mode="Train", target_idx=None):
        self.suf_dis = []
        self.prev_state = None
        self.obstacle = AABB([-165, -470, 120])
        self.min_dis = np.inf
        if max_step:
            self.max_step = max_step

        if target_points is not None:
            self.target_points = target_points

        # reset
        if mode == "Train":  # add noise
            self.target_idx = np.random.choice(np.arange(8), p=None)
            if self.add_noise:
                v_noise = np.random.randn(6) * 0.002
                t_noise = np.random.randn(3) * 5
            else:
                v_noise = np.zeros(6)
                t_noise = np.zeros(3)
        else:
            self.target_idx = np.random.randint(self.target_points.shape[0])
            v_noise = np.zeros(6)
            t_noise = np.zeros(3)

        if self.control_mode == "continuous":
            self.target_idx = 0

        if target_idx is not None:
            self.target_idx = target_idx

        self.prev_vel = np.zeros(6)
        self.acc = np.zeros(6)
        self.robot.q = self.initial_poses[self.target_idx] + v_noise

        p = self.robot.fkine_all(self.robot.q)
        robot_pos = np.zeros((6, 3), dtype=float)
        for i in range(1, 7):
            robot_pos[i - 1, :] = p[i].t * 1000.0

        self.cur_step = 0

        self.target = self.target_points[self.target_idx] + t_noise
        self.prev_target_dis = np.linalg.norm(robot_pos[-1, :] - self.target) / 1000.0
        self.initial_dis = self.prev_target_dis

        print(f"Init Joint6 Pos {robot_pos[-1, :]} Target {self.target_idx} Pos {self.target} "
              f"Dis_targ {self.prev_target_dis}")

        if self.detect_collision:
            self.traj_idx = np.random.randint(self.human_traj_data.shape[0])
            if not self.loop_human:
                human_joints_pos = traj_interpolate(self.human_traj_data[self.traj_idx], self.human_time_list,
                                (self.target_idx * self.single_max_step + self.cur_step) * self.single_step_duration)
            else:
                human_joints_pos = traj_interpolate(self.human_traj_data[self.traj_idx], self.human_time_list,
                                        (self.global_step % self.human_time_list.shape[0]) * self.single_step_duration)

            if self.add_noise:
                human_joints_pos += np.random.randn(human_joints_pos.shape[0], human_joints_pos.shape[1]) * 10
            self.update_collider(self.robot.q, p, robot_pos, human_joints_pos)
            self.prev_link_dis, _, _ = self.collision_detection()
            if self.obj_type == "human":
                next_sate = np.concatenate([self.prev_vel, robot_pos.flatten(), self.target,
                                            human_joints_pos.flatten()])

        else:
            self.prev_link_dis = [np.inf, np.inf, np.inf, np.inf]
            next_sate = np.concatenate([self.prev_vel, robot_pos.flatten(), self.target])

        self.cur_state = next_sate

        return next_sate

    def step(self, action):
        """
        :param action: 6 joints angular velocity
        :return:
        """
        # take an action
        ang_vel = action
        self.robot.q = self.robot.q + ang_vel * self.single_step_duration
        self.acc = ang_vel - self.prev_vel
        self.prev_vel = ang_vel

        p = self.robot.fkine_all(self.robot.q)
        next_robot_pos = np.zeros((6, 3))
        for i in range(1, 7):
            next_robot_pos[i - 1, :] = p[i].t * 1000

        self.cur_step += 1

        if self.detect_collision:
            if not self.loop_human:
                human_joints_pos = traj_interpolate(self.human_traj_data[self.traj_idx], self.human_time_list,
                                (self.target_idx * self.single_max_step + self.cur_step) * self.single_step_duration)
            else:
                self.global_step += 1
                if self.global_step % self.human_time_list.shape[0] == 0:
                    self.traj_idx = np.random.randint(self.human_traj_data.shape[0])
                human_joints_pos = traj_interpolate(self.human_traj_data[self.traj_idx], self.human_time_list,
                                    (self.global_step % self.human_time_list.shape[0]) * self.single_step_duration)

            if self.add_noise:
                human_joints_pos += np.random.randn(human_joints_pos.shape[0], human_joints_pos.shape[1]) * 10
            self.update_collider(self.robot.q, p, next_robot_pos, human_joints_pos)
            if self.obj_type == "human":
                next_sate = np.concatenate([action, next_robot_pos.flatten(), self.target,
                                            human_joints_pos.flatten()])

        else:
            next_sate = np.concatenate([action, next_robot_pos.flatten(), self.target])



        self.prev_state = self.cur_state
        self.cur_state = next_sate
        r, dis_targ, s_suc, fail = self.reward(next_sate)


        suc = False
        done = False
        if s_suc:
            print("Step %d Target %d Success!" % (self.cur_step, self.target_idx))
            print(f"Cur Joint6 Pos{next_robot_pos[-1, :]} Target {self.target_idx} "
                  f"Pos {self.target} Dis_targ {dis_targ}")

            if self.control_mode == "segmental":
                done = True
                suc = True

        elif fail:  # 发生碰撞终止
            done = True  # 结束回合
            print(f"Step {self.cur_step} Collision Occur! Link Dis {self.prev_link_dis} Obs Pos{self.obstacle.center}")
            print(f"Final Joint6 Pos{next_robot_pos[-1, :]} Target {self.target_idx} Pos {self.target} Dis_targ {dis_targ}")

        if self.cur_step == self.max_step and not suc and not fail:  # 最大步数终止
            done = True
            print(f"Reaching Max steps! Final Joint6 Pos{next_robot_pos[-1, :]} Target {self.target_idx} Pos {self.target} Dis_targ {dis_targ}")

        sf = suc or fail

        return next_sate, r, done, sf, suc
