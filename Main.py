import os
import time
import numpy as np
import torch
from roboticstoolbox import jtraj
from torch.utils.tensorboard import SummaryWriter
from DDPG import DDPG
from Environment import HREnv
from FR5 import FR5
from SAC import SACAgent
from TD3 import TD3
from Utils import *
from Plots import *

global reward_scaling

# Agent Params
Train_Episodes = 100000
Test_Episodes = 8 * 1
Val_Episodes = 50

Use_Reward_Scaling = False
Use_State_Normalization = False

# ERE Params
Use_ERE = True
Eta_0 = 0.996
Eta_T = 1.0
C_k_min = 2500

Use_Suc_Pool = False

Save_Interval = 500
Val_Interval = 500

Initial_Poses = np.array([[28.895, -104.803, 129.158, -114.355, -90, -16.105],
                          [28.895, -109.767, 110.049, -90.279, -90, -16.105],
                          [86.589, -83.919, 94.809, -100.888, -90, 41.589],
                          [86.589, -82, 100.887, -108.887, -90, 41.589],
                          [86.589, -83.919, 94.809, -100.888, -90, 41.589],
                          [34.608, -81.066, 91.545, -100.479, -90, -10.388],
                          [34.608, -64.85, 110.969, -136.124, -90, -10.388],
                          [34.608, -81.066, 91.545, -100.479, -90, -10.388]])
Initial_Poses = np.deg2rad(Initial_Poses)

Target_Points = np.array([[-260, -260, 450],
                          [70, -540, 400],
                          [70, -540, 345],
                          [70, -540, 400],
                          [-400, -400, 400],
                          [-400, -400, 152],
                          [-400, -400, 400],
                          [-260, -260, 300]])

Target_Num = 8
Episode_Maximum_Step = 50
Single_Max_Step = Episode_Maximum_Step
Update_Episode_Frequency = 20


data_path = "./human_traj_data/assemble_train_data_new.npy"
pt_path = "./save_models/ppo2/epi99999.pt"
a2c_apt = "./save_models/a2c-human-ds0.2-ov0.05/aepi5999.pt"
a2c_cpt = "./save_models/a2c-human-ds0.2-ov0.05/cepi5999.pt"
ppo_apt = "./save_models/ppo-human-ds0.2-ov0.05/aepi5999.pt"
ppo_cpt = "./save_models/ppo-human-ds0.2-ov0.05/cepi5999.pt"
sac_a_pt = "./save_models/sac-human2-ere-dc2-ds0.1-ov0.05/aepi49999.pt"
sac_c_pt = "./save_models/sac-human2-ere-dc2-ds0.1-ov0.05/cepi49999.pt"
ddpg_a_pt = "./save_models/DDPG-human-ds0.2/aepi49999.pt"
ddpg_at_pt = "./save_models/DDPG-human-ds0.2/atepi49999.pt"
ddpg_c_pt = "./save_models/DDPG-human-ds0.2/cepi49999.pt"
ddpg_ct_pt = "./save_models/DDPG-human-ds0.2/ctepi49999.pt"
td3_a_pt = "./save_models/TD3-human2-ds0.2/aepi49999.pt"
td3_at_pt = "./save_models/TD3-human2-ds0.2/atepi49999.pt"
td3_c_pt = "./save_models/TD3-human2-ds0.2/cepi49999.pt"
td3_ct_pt = "./save_models/TD3-human2-ds0.2/ctepi49999.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = HREnv(Target_Points, Initial_Poses, data_path, 50, detect_collision=True, control_mode="segmental",
            obj_type="human", add_noise=True, loop_human=False)
agent_sac = SACAgent(env.state_Dim, env.action_Dim, 128, np.deg2rad(180), device, use_ERE=Use_ERE,
                     use_suc_pool=Use_Suc_Pool, mask=False)
agent_ddpg = DDPG(env.state_Dim, env.action_Dim, np.deg2rad(180), 128, device)
agent_td3 = TD3(env.state_Dim, env.action_Dim, np.deg2rad(180), device)

# torch.autograd.set_detect_anomaly(True)


def main_sac(mode="train", resume=False):
    exp_name = "SAC-train/"
    model_path = "save_models/" + exp_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if mode == "train":
        writer = SummaryWriter(log_dir=model_path, comment=exp_name)
        if resume:
            start_epi = int(sac_a_pt[-8:-3]) + 1
            print(start_epi)
            agent_sac.load(sac_a_pt, sac_c_pt)
        else:
            start_epi = 0

        start_time = time.time()
        iter_num = 0

        for epi in range(start_epi, start_epi + Train_Episodes):
            agent_sac.eval_prep()
            cur_state = env.reset(target_idx=1)
            episode_reward = 0
            a_loss = 0
            c_loss = 0
            alpha_loss = 0
            for step in range(env.max_step):
                iter_num += 1
                a_tanh, action = agent_sac.choose_action(torch.tensor(cur_state, dtype=torch.float).to(device))
                next_state, r, done, sf, suc = env.step(action)

                episode_reward += r

                agent_sac.replay_buffer.store(cur_state, a_tanh, r, next_state, done, env.target_idx)
                eta_t = Eta_0 + (Eta_T - Eta_0) * (iter_num / (Train_Episodes * Single_Max_Step))
                cur_state = next_state

                if done:
                    writer.add_scalar("success", int(suc), epi)
                    agent_sac.train_prep()
                    if Use_Suc_Pool:
                        if suc:
                            agent_sac.replay_buffer.add_cur_traj_to_suc_pool()
                        else:
                            agent_sac.replay_buffer.clear_cur_traj()
                    a_losses = []
                    c_losses = []
                    alpha_losses = []

                    for k in range(1, step + 1):
                        if Use_ERE:
                            c_k = max(
                                int(agent_sac.replay_buffer.__len__() * eta_t ** (k * (env.max_step / (step + 1)))),
                                C_k_min)
                        else:
                            c_k = None
                        a_loss_, c_loss_, alpha_loss_ = agent_sac.update(c_k)
                        a_losses.append(a_loss_)
                        c_losses.append(c_loss_)
                        alpha_losses.append(alpha_loss_)
                    a_loss = np.mean(a_losses)
                    c_loss = np.mean(c_losses)
                    alpha_loss = np.mean(alpha_losses)
                    print(action)
                    print(f"a_loss:{a_loss}, c_loss:{c_loss}， alpha_loss:{alpha_loss}")

                if sf:  # 到达最后路径点或发生碰撞结束epoch
                    break

            if (epi + 1) % Save_Interval == 0:
                torch.save(agent_sac.actor.state_dict(), model_path + f"aepi{epi}.pt")
                torch.save(agent_sac.critic_target.state_dict(), model_path + f"cepi{epi}.pt")

            if (epi + 1) % Val_Interval == 0:
                agent_sac.eval_prep()
                success_num = 0
                val_rewards = []
                for v_epi in range(Val_Episodes):
                    cur_state = env.reset(mode='test', target_idx=1)
                    v_episode_reward = 0
                    for step in range(env.max_step):
                        a_tanh, action = agent_sac.choose_action(torch.tensor(cur_state, dtype=torch.float).to(device),
                                                                 True)
                        next_state, r, done, sf, suc = env.step(action)
                        v_episode_reward += r
                        cur_state = next_state

                        if done:
                            val_rewards.append(v_episode_reward)
                        if suc:
                            success_num += 1
                        if sf:
                            break
                success_rate = success_num / Val_Episodes
                print("Eval Avg Reward: %f Success Rate %f" % (np.mean(val_rewards), success_rate))
                writer.add_scalar("val_success_rate", success_rate, (epi + 1) // Val_Interval)
                writer.add_scalar("val_avg_rewards", np.mean(val_rewards), (epi + 1) // Val_Interval)

            writer.add_scalar("target%d/a_loss" % env.target_idx, a_loss, epi)
            writer.add_scalar("target%d/c_loss" % env.target_idx, c_loss, epi)
            writer.add_scalar("target%d/alpha_loss" % env.target_idx, alpha_loss, epi)
            writer.add_scalar("target%d/epi_rewards" % env.target_idx, episode_reward, epi)
            writer.add_scalar("a_loss", a_loss, epi)
            writer.add_scalar("c_loss", c_loss, epi)
            writer.add_scalar("alpha_loss", alpha_loss, epi)
            writer.add_scalar("epi_rewards", episode_reward, epi)

            print("Train Episode %d done reward %f" % (epi, episode_reward))

        end_time = time.time()
        sec = end_time - start_time
        print("all train episode time cost %d min %d s" % (sec // 60, sec % 60))


def main_ddpg_TD3(agent, mode="train", resume=False):
    assert isinstance(agent, DDPG) or isinstance(agent, TD3)
    exp_name = agent.__class__.__name__ + "-train/"
    random_episodes = 500
    model_path = "./save_models/" + exp_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if isinstance(agent, DDPG):
        a_pt = ddpg_a_pt
        at_pt = ddpg_at_pt
        c_pt = ddpg_c_pt
        ct_pt = ddpg_ct_pt
        b_deterministic = False
    elif isinstance(agent, TD3):
        a_pt = td3_a_pt
        at_pt = td3_at_pt
        c_pt = td3_c_pt
        ct_pt = td3_ct_pt
        b_deterministic = True

    if mode == "train":
        writer = SummaryWriter(log_dir=model_path, comment=exp_name)
        if resume:
            start_epi = int(a_pt[-8:-3]) + 1
            print(start_epi)
            agent.load(a_pt, at_pt, c_pt, ct_pt)
        else:
            start_epi = 0

        start_time = time.time()
        iter_num = 0

        for epi in range(start_epi, start_epi + Train_Episodes):
            agent.eval_prep()
            cur_state = env.reset()
            episode_reward = 0
            a_loss = 0
            c_loss = 0

            for step in range(env.max_step):
                iter_num += 1
                # 得到rad角速度
                if epi < random_episodes:
                    a = np.random.uniform(-1, 1, 6)
                    action = a * agent.max_action

                else:
                    action, a = agent.choose_action(torch.tensor(cur_state, dtype=torch.float).to(device),
                                                    deterministic=b_deterministic)
                next_state, r, done, sf, suc = env.step(action)
                episode_reward += r

                agent.replay_buffer.store(cur_state, action, r, next_state, sf, env.target_idx)
                cur_state = next_state

                if done:
                    writer.add_scalar("success", int(suc), epi)
                    agent.train_prep()
                    if suc and Use_Suc_Pool:
                        agent.replay_buffer.add_cur_traj_to_suc_pool()
                    else:
                        agent.replay_buffer.clear_cur_traj()
                    a_losses = []
                    c_losses = []
                    for k in range(1, step + 1):
                        a_loss_, c_loss_ = agent.update()
                        a_losses.append(a_loss_)
                        c_losses.append(c_loss_)
                    a_loss = np.mean(a_losses)
                    c_loss = np.mean(c_losses)
                    print(f"a_loss:{a_loss}, c_loss:{c_loss}")

                if sf:  # 到达最后路径点或发生碰撞结束epoch
                    break

            if (epi + 1) % Save_Interval == 0:
                torch.save(agent.actor.state_dict(), model_path + f"aepi{epi}.pt")
                torch.save(agent.actor_target.state_dict(), model_path + f"atepi{epi}.pt")
                torch.save(agent.critic.state_dict(), model_path + f"cepi{epi}.pt")
                torch.save(agent.critic_target.state_dict(), model_path + f"ctepi{epi}.pt")

            if (epi + 1) % Val_Interval == 0:
                agent.eval_prep()
                success_num = 0
                val_rewards = []
                for v_epi in range(Val_Episodes):
                    cur_state = env.reset(mode='test')
                    v_episode_reward = 0
                    for step in range(env.max_step):
                        action, a = agent.choose_action(torch.tensor(cur_state, dtype=torch.float).to(device),
                                                        deterministic=True)
                        next_state, r, done, sf, suc = env.step(action)
                        v_episode_reward += r
                        cur_state = next_state

                        if done:
                            val_rewards.append(v_episode_reward)
                        if suc:
                            success_num += 1
                        if sf:
                            break
                success_rate = success_num / Val_Episodes
                print("Eval Avg Reward: %f Success Rate %f" % (np.mean(val_rewards), success_rate))
                writer.add_scalar("val_success_rate", success_rate, (epi + 1) // Val_Interval)
                writer.add_scalar("val_avg_rewards", np.mean(val_rewards), (epi + 1) // Val_Interval)

            writer.add_scalar("target%d/a_loss" % env.target_idx, a_loss, epi)
            writer.add_scalar("target%d/c_loss" % env.target_idx, c_loss, epi)
            writer.add_scalar("target%d/epi_rewards" % env.target_idx, episode_reward, epi)
            writer.add_scalar("a_loss", a_loss, epi)
            writer.add_scalar("c_loss", c_loss, epi)
            writer.add_scalar("epi_rewards", episode_reward, epi)

            print("Train Episode %d done reward %f" % (epi, episode_reward))

        end_time = time.time()
        sec = end_time - start_time
        print("all train episode time cost %d min %d s" % (sec // 60, sec % 60))


def test_agent(agent, save_path, exp_name, a_pt, c_pt, at_pt=None, ct_pt=None, post_process=False, save_fig=False):
    if at_pt is None:
        agent.load(a_pt, c_pt)
    else:
        agent.load(a_pt, c_pt, at_pt, ct_pt)
    agent.eval_prep()
    success_num = 0
    test_rewards = []
    q_traj = []
    j6_pos = []
    min_suf_dis = np.zeros(8)
    mean_suf_dis = np.zeros(8)
    time_list = np.array([i * 0.05 for i in range(5)])
    st = time.time()
    for epi in range(Test_Episodes):
        # env.control_mode = "continuous"
        cur_state = env.reset(mode='test', target_idx=epi % 8)
        q_traj.append(np.expand_dims(env.robot.q, 0))
        j6_pos.append(cur_state[21:24])
        episode_reward = 0
        for step in range(env.max_step):
            if isinstance(agent, SACAgent):
                a_tanh, action = agent.choose_action(torch.tensor(cur_state, dtype=torch.float).to(device), True)
            else:
                action, a = agent.choose_action(torch.tensor(cur_state, dtype=torch.float).to(device), True)

            next_state, r, done, sf, suc = env.step(action)
            q = np.expand_dims(env.robot.q, 0)
            q_traj.append(q)

            episode_reward += r
            cur_state = next_state
            j6_pos.append(cur_state[21:24])

            if done:
                test_rewards.append(episode_reward)
                print("test epi %d min_dis %.5f" % (epi, env.min_dis))
                min_suf_dis[epi % 8] += env.min_dis
                mean_suf_dis[epi % 8] += np.mean(env.suf_dis)
            if suc:
                success_num += 1
                break
            if sf:
                break
        if post_process:
            tmp_traj = jtraj(env.robot.q, Initial_Poses[(env.target_idx + 1) % 8], time_list)
            q_traj.append(tmp_traj.q)

    success_rate = success_num / Test_Episodes
    if post_process:
        np.save(save_path + f"q_traj_pp.npy", np.concatenate(q_traj))
        np.savetxt(save_path + f"q_traj_pp.lua", np.rad2deg(np.concatenate(q_traj)), fmt="%f", delimiter=",")
    else:
        np.save(save_path + f"q_traj.npy", np.concatenate(q_traj))
        np.savetxt(save_path + f"q_traj.lua", np.concatenate(q_traj), delimiter=",")

    np.save(save_path + f"qj6_pos.npy", np.array(j6_pos))
    min_suf_dis = min_suf_dis / (Test_Episodes // 8)
    mean_suf_dis = mean_suf_dis / (Test_Episodes // 8)
    print("Test Avg Reward: %f Success Rate %f" % (np.mean(test_rewards), success_rate))
    print("Test Avg min suf dis \n", min_suf_dis, np.mean(min_suf_dis))
    print("Test Avg mean suf dis \n", mean_suf_dis, mean_suf_dis.mean())
    # env.robot.plot(np.array(q_traj), dt=0.05, block=True, loop=True)
    draw_result_path(np.array(j6_pos), exp_name[:-1], save=save_fig)
    print("test %d episodes time cost %.4f" % (Test_Episodes, time.time() - st))


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    td3_exp_name = "TD3"
    td3_save_path = "./save_models/" + td3_exp_name + "/"
    sac_exp_name = "SAC"
    sac_save_path = "./save_models/" + sac_exp_name + "/"
    model_epi_num = 99999

    sac_a_pt = sac_save_path + "aepi{epi}.pt".format(epi=model_epi_num)
    sac_c_pt = sac_save_path + "cepi{epi}.pt".format(epi=model_epi_num)
    td3_a_pt = td3_save_path + "aepi{epi}.pt".format(epi=model_epi_num)
    td3_at_pt = td3_save_path + "atepi{epi}.pt".format(epi=model_epi_num)
    td3_c_pt = td3_save_path + "cepi{epi}.pt".format(epi=model_epi_num)
    td3_ct_pt = td3_save_path + "ctepi{epi}.pt".format(epi=model_epi_num)

    main_sac("train", False)
    # main_ddpg_TD3(agent_td3, "train")
    # test_agent(agent_td3, td3_save_path, td3_exp_name, td3_a_pt, td3_c_pt, td3_at_pt, td3_ct_pt, True, False)
    test_agent(agent_sac, sac_save_path, sac_exp_name, sac_a_pt, sac_c_pt, post_process=True, save_fig=False)

