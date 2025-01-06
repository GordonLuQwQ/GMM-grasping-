from unittest import TestCase

import numpy as np
import pickle
from spatialmath import SE3, SO3
import mujoco
import mujoco.viewer

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from imitation_learning.gaussian_mixture_model import GMM
from src.motion_planning import *
from src.robot import IIWA14


class TestGMM(TestCase):

    def test_gmm(self):
        X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

        gmm = GaussianMixture(n_components=4)
        gmm.fit(X)

        labels = gmm.predict(X)

        plt.figure(1)
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', alpha=0.6)
        plt.colorbar()
        plt.title('GMM Clustering')
        plt.show()

    def test_generate_data(self):
        model = mujoco.MjModel.from_xml_path("../assets/kuka_iiwa_14/scene.xml")
        data = mujoco.MjData(model)

        mujoco.mj_resetData(model, data)
        q0 = np.zeros(model.nq)
        q0[:7] = [0.0, np.pi / 4, 0.0, -np.pi / 4, 0.0, np.pi / 2, 0.0]
        mujoco.mj_setState(model, data, q0, mujoco.mjtState.mjSTATE_QPOS)
        mujoco.mj_forward(model, data)

        data.mocap_pos[0, :] = [np.random.random() * 0.3 + 0.5, np.random.random() * 0.6 - 0.3,
                                np.random.random() * 0.3 + 0.1]

        robot = IIWA14(tool=np.array([0.0, 0.0, 0.1488]))
        robot.set_joint(q0[:7])
        T0 = robot.fkine(q0[:7])
        Te = T0
        gripper_joint = 0.0

        motion_time = 2.0

        time0 = motion_time

        time1 = motion_time
        t0 = T0.t
        R0 = SO3.Ry(np.pi)
        t1 = data.mocap_pos[0, :] + [0.0, 0.0, 0.05]
        R1 = R0
        position_parameter1 = LinePositionParameter(t0, t1)
        attitude_parameter1 = OneAttitudeParameter(R0, R1)
        cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1)
        velocity_parameter1 = QuinticVelocityParameter(time1)
        trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1)
        trajectory_planner1 = TrajectoryPlanner(trajectory_parameter1)

        time2 = motion_time
        t2 = data.mocap_pos[0, :]
        R2 = R1
        position_parameter2 = LinePositionParameter(t1, t2)
        attitude_parameter2 = OneAttitudeParameter(R1, R2)
        cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
        velocity_parameter2 = QuinticVelocityParameter(time2)
        trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
        trajectory_planner2 = TrajectoryPlanner(trajectory_parameter2)

        time3 = motion_time

        times = np.array([time0, time1, time2, time3])
        trajectory_planners = [trajectory_planner1, trajectory_planner2]

        collect_data = []
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():

                if data.time > 10:
                    break

                if time0 <= data.time < np.sum(times):

                    if data.time < np.sum(times[:-1]):
                        for i in range(1, times.size - 1):
                            if data.time < np.sum(times[: i + 1]):
                                Te = trajectory_planners[i - 1].interpolate(data.time - np.sum(times[: i]))
                                gripper_joint = 0.0
                                break
                    else:
                        gripper_joint = (data.time - np.sum(times[: -1])) / times[-1] * 255

                    collect_data.append(np.hstack((data.time - times[0], Te.t, gripper_joint)))

                robot.move_cartesian(Te)
                qe = robot.get_joint()

                ctrl = np.hstack((qe, gripper_joint))
                mujoco.mj_setState(model, data, ctrl, mujoco.mjtState.mjSTATE_CTRL)

                mujoco.mj_step(model, data)

                viewer.sync()

        np.savetxt('./collect_data1.csv', np.array(collect_data), delimiter=',')

    def test_gmm_train(self):
        collect_data = np.array([])
        ps = []
        for i in range(10):
            data_i = np.genfromtxt('./collect_data' + str(i + 1) + '.csv', delimiter=',')
            A0 = np.eye(5)

            n = (data_i[1, 1:4] - data_i[0, 1:4]) / np.linalg.norm(data_i[1, 1:4] - data_i[0, 1:4])
            a = np.array([0, 0, 1])
            o = np.cross(a, n) / np.linalg.norm(np.cross(a, n))
            a = np.cross(n, o)
            A0[1:4, 1:4] = np.vstack((n, o, a)).T

            b0 = data_i[0, :]
            A1 = np.eye(5)
            b1 = data_i[-1, :]
            b1[0] = 0
            p = [[A0, b0], [A1, b1]]
            ps.append(p)
            if i == 0:
                collect_data = data_i
            else:
                collect_data = np.hstack((collect_data, data_i))
        collect_data = collect_data.T

        nb_data = 3001
        data = np.zeros((5, 2, nb_data * 10))
        for n in range(10):
            for m in range(2):
                data[:, m, n * nb_data: (n + 1) * nb_data] = np.linalg.inv(ps[n][m][0]) @ (
                        collect_data[n * 5: (n + 1) * 5, :].T - ps[n][m][1]).T
        nb_states = 8
        nb_frames = 2
        nb_var = 5
        gmm = GMM(nb_states, nb_frames, nb_var)
        gmm.train(data)

        with open('./gmm_model2.pkl', 'wb') as file:
            pickle.dump(gmm, file)

    def test_gmm_reproduce(self):
        with open('./gmm_model.pkl', 'rb') as file:
            gmm = pickle.load(file)

        model = mujoco.MjModel.from_xml_path("../assets/kuka_iiwa_14/scene.xml")
        data = mujoco.MjData(model)

        q0_robot = [0.0, np.pi / 4, 0.0, -np.pi / 4, 0.0, np.pi / 2, 0.0]
 
        robot = IIWA14(tool=np.array([0.0, 0.0, 0.1488]))
        robot.set_joint(q0_robot)
        T0 = robot.fkine(q0_robot)
   
        mujoco.mj_resetData(model, data)
        q0 = np.zeros(model.nq)
        q0[:7] = q0_robot
        mujoco.mj_setState(model, data, q0, mujoco.mjtState.mjSTATE_QPOS)
        mujoco.mj_forward(model, data)

        start = T0.t
        #goal = np.array(
            #[np.random.random() * 0.3 + 0.4, np.random.random() * 0.6 - 0.3, np.random.random() * 0.3 + 0.1])
        goal = np.array(
            [0.1 * 0.28 + 0.4, 0.29 * 0.6 - 0.3, 0.29 * 0.3 + 0.1])
        data.mocap_pos[0, :] = goal
        print(start)
        print(goal)
        #
        reproduce_trajectory = gmm.reproduce(start=start, goal=goal)
        collect_data = []
        time_num = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():

                Te = SE3(reproduce_trajectory[:3, time_num]) * SE3(SO3(T0.R))
                robot.move_cartesian(Te)
                qe = robot.get_joint()

                gripper_joint = reproduce_trajectory[3, time_num]
                ctrl = np.hstack((qe, gripper_joint))

                mujoco.mj_setState(model, data, ctrl, mujoco.mjtState.mjSTATE_CTRL)

                mujoco.mj_step(model, data)

                if data.time > 2.0 and time_num < 3000:
                    time_num += 1
                current_data = np.hstack((reproduce_trajectory[:, time_num], [time_num]))
                collect_data.append(current_data)
                viewer.sync()
        collect_data_array = np.array(collect_data)
        np.savetxt('./reproduce_trajectory_with_time.csv', collect_data_array, delimiter=',', header='x, y, z, gripper, time_num', comments='')
if __name__ == "__main__":
    test_instance = TestGMM()
    for i in range(1):  
        test_instance.test_gmm_reproduce()
 