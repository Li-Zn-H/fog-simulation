import os
import math
import copy
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.constants import speed_of_light as c

data = np.random.normal(size=[100, 4])
RNG = np.random.default_rng(seed=42)  # 随机数种子
AVAILABLE_TAU_Hs = [20]  # 半脉冲宽度
LIDAR_FOLDERS = ['lidar_hdl64_strongest', 'lidar_hdl64_last']  # 传感器文件夹
INTEGRAL_PATH = Path(os.path.dirname(os.path.realpath(__file__))) / 'integral_lookup_tables' / 'original'  # 临时变量存储路径


class ParameterSet:
    def __init__(self, **kwargs) -> None:
        self.n = 500
        self.n_min = 100
        self.n_max = 1000

        self.r_range = 100
        self.r_range_min = 50
        self.r_range_max = 250

        # 衰减系数 => 雾量
        self.alpha = 0.06
        self.alpha_min = 0.003
        self.alpha_max = 0.5
        self.alpha_scale = 1000

        # 气象光学范围（米）
        self.mor = np.log(20) / self.alpha

        # 向后散射系数（in 1/sr）[sr = 球面度]
        self.beta = 0.046 / self.mor
        self.beta_min = 0.023 / self.mor
        self.beta_max = 0.092 / self.mor
        self.beta_scale = 1000 * self.mor

        # 脉冲峰值功率（瓦）
        self.p_0 = 80
        self.p_0_min = 60
        self.p_0_max = 100

        # 半功率脉冲宽度（秒）
        self.tau_h = 2e-8
        self.tau_h_min = 5e-9
        self.tau_h_max = 8e-8
        self.tau_h_scale = 1e9

        # 总脉冲能力（焦）
        self.e_p = self.p_0 * self.tau_h  # equation (7) in [1]

        # 接收器孔径面积（米^2）
        self.a_r = 0.25
        self.a_r_min = 0.01
        self.a_r_max = 0.1
        self.a_r_scale = 1000

        # 接收器光学损失
        self.l_r = 0.05
        self.l_r_min = 0.01
        self.l_r_max = 0.10
        self.l_r_scale = 100

        self.c_a = c * self.l_r * self.a_r / 2

        self.linear_xsi = True

        self.D = 0.1  # 发射器和接收器的位移 米
        self.ROH_T = 0.01  # 发射器孔径半径 米
        self.ROH_R = 0.01  # 接收器孔径半径 米
        self.GAMMA_T_DEG = 2  # 发射器视野的开启角度 角度
        self.GAMMA_R_DEG = 3.5  # 接收器事业的开启角度 角度
        self.GAMMA_T = math.radians(self.GAMMA_T_DEG)
        self.GAMMA_R = math.radians(self.GAMMA_R_DEG)

        # 接收器视场开始覆盖发射波束的范围（米）
        self.r_1 = 0.9
        self.r_1_min = 0
        self.r_1_max = 10
        self.r_1_scale = 10

        # 接收器视场完全覆盖发射波束的范围（米）
        self.r_2 = 1.0
        self.r_2_min = 0
        self.r_2_max = 10
        self.r_2_scale = 10

        # 到硬目标的距离（m）
        self.r_0 = 30
        self.r_0_min = 1
        self.r_0_max = 200

        # 硬目标的反射率[0.07，0.2， > 4 = > 低，正常，高]
        self.gamma = 0.000001
        self.gamma_min = 0.0000001
        self.gamma_max = 0.00001
        self.gamma_scale = 10000000

        # 目标的微分反射率
        self.beta_0 = self.gamma / np.pi

        self.__dict__.update(kwargs)


def get_available_alphas() -> List[float]:  # 得到所有文件的阿尔法值参数
    alphas = []
    for file in os.listdir(INTEGRAL_PATH):  # 返回指定的文件夹包含的文件或文件夹的名字的列表。
        if file.endswith(".pickle"):
            alpha = file.split('_')[-1].replace('.pickle', '')
            alphas.append(float(alpha))
    return sorted(alphas)


def get_integral_dict(p: ParameterSet) -> Dict:
    alphas = get_available_alphas()
    alpha = min(alphas, key=lambda x: abs(x - p.alpha))
    tau_h = min(AVAILABLE_TAU_Hs, key=lambda x: abs(x - int(p.tau_h * 1e9)))

    filename = INTEGRAL_PATH / f'integral_0m_to_200m_stepsize_0.1m_tau_h_{tau_h}ns_alpha_{alpha}.pickle'

    with open(filename, 'rb') as handle:
        integral_dict = pickle.load(handle)
        print(integral_dict)
    return integral_dict


def P_R_fog_hard(p: ParameterSet, pc: np.ndarray) -> np.ndarray:
    r_0 = np.linalg.norm(pc[:, 0:3], axis=1)  # 求距离R_0
    pc[:, 3] = np.round(np.exp(-2 * p.alpha * r_0) * pc[:, 3])  # 求硬点云数据
    return pc


def P_R_fog_soft(p: ParameterSet, pc: np.ndarray, original_intesity: np.ndarray, noise: int, gain: bool = False,
                 noise_variant: str = 'v1') -> Tuple[np.ndarray, np.ndarray, Dict]:
    augmented_pc = np.zeros(pc.shape)  # 复制点云形状
    fog_mask = np.zeros(len(pc), dtype=bool)  # 制作假标签

    r_zeros = np.linalg.norm(pc[:, 0:3], axis=1)  # 求点云距离R_0

    min_fog_response = np.inf
    max_fog_response = 0
    num_fog_responses = 0

    integral_dict = get_integral_dict(p)

    r_noise = RNG.integers(low=1, high=20, size=1)[0]

    for i, r_0 in enumerate(r_zeros):  # 依次遍历每个点云（下标，距离）
        # 从预先计算的字典中加载积分值
        key = float(str(round(r_0, 1)))
        # 限制键大小最大为200
        fog_distance, fog_response = integral_dict[min(key, 200)]  # 计算雾距离和反射

        fog_response = fog_response * original_intesity[i] * (r_0 ** 2) * p.beta / p.beta_0  # 计算i_soft

        # 限制在255以内
        fog_response = min(fog_response, 255)

        if fog_response > pc[i, 3]:  # 如果雾的反射率比目标大
            fog_mask[i] = 1  # 标记该点
            num_fog_responses += 1  # 软目标反射率总数加一

            scaling_factor = fog_distance / r_0  # 计算缩放因子

            # 缩放后的点云数据
            augmented_pc[i, 0] = pc[i, 0] * scaling_factor
            augmented_pc[i, 1] = pc[i, 1] * scaling_factor
            augmented_pc[i, 2] = pc[i, 2] * scaling_factor
            augmented_pc[i, 3] = fog_response

            # 如果有第五个特征则直接保存
            if pc.shape[1] > 4:
                augmented_pc[i, 4] = pc[i, 4]

            if noise > 0:  # 如果有噪声

                if noise_variant == 'v1':  # 第一类噪声种类

                    # 基于距离添加统一的噪声
                    distance_noise = RNG.uniform(low=r_0 - noise, high=r_0 + noise, size=1)[0]
                    noise_factor = r_0 / distance_noise

                elif noise_variant == 'v2':  # 第二类噪声种类

                    # add noise in the power domain
                    power = RNG.uniform(low=-1, high=1, size=1)[0]
                    noise_factor = max(1.0, noise / 5) ** power  # noise=10 => noise_factor ranges from 1/2 to 2

                elif noise_variant == 'v3':  # 第三类噪声种类

                    # add noise in the power domain
                    power = RNG.uniform(low=-0.5, high=1, size=1)[0]
                    noise_factor = max(1.0, noise * 4 / 10) ** power  # noise=10 => ranges from 1/2 to 4

                elif noise_variant == 'v4':  # 第四类噪声种类

                    additive = r_noise * RNG.beta(a=2, b=20, size=1)[0]  # β分布是一种迪利克雷分布
                    new_dist = fog_distance + additive
                    noise_factor = new_dist / fog_distance

                else:  # 异常

                    raise NotImplementedError(f"noise variant '{noise_variant}' is not implemented (yet)")

                augmented_pc[i, 0] = augmented_pc[i, 0] * noise_factor  # x
                augmented_pc[i, 1] = augmented_pc[i, 1] * noise_factor  # y
                augmented_pc[i, 2] = augmented_pc[i, 2] * noise_factor  # z

            if fog_response > max_fog_response:  # 找最大的反射点
                max_fog_response = fog_response

            if fog_response < min_fog_response:  # 找最小的反射点
                min_fog_response = fog_response

        else:  # 硬目标

            augmented_pc[i] = pc[i]

    if gain:  # 传入的参数gain
        max_intensity = np.ceil(max(augmented_pc[:, 3]))  # 最大强度 = 向上取整(最大的反射值)
        gain_factor = 255 / max_intensity
        augmented_pc[:, 3] *= gain_factor  # 增强点云

    simulated_fog_pc = None

    if num_fog_responses > 0:  # 如果数目大于0
        fog_points = augmented_pc[fog_mask]  # 找出雾点
        simulated_fog_pc = fog_points

    info_dict = {'min_fog_response': min_fog_response,
                 'max_fog_response': max_fog_response,
                 'num_fog_responses': num_fog_responses}

    # 得到一些内容信息
    return augmented_pc, simulated_fog_pc, info_dict


def simulate_fog(p: ParameterSet, pc: np.ndarray, noise: int, gain: bool = False, noise_variant: str = 'v1',
                 hard: bool = True, soft: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:

    augmented_pc = copy.deepcopy(pc)    # 增强的点云数据 = 点云的复制
    original_intensity = copy.deepcopy(pc[:, 3])    # 原来的xyz

    info_dict = None
    simulated_fog_pc = None

    if hard:
        augmented_pc = P_R_fog_hard(p, augmented_pc)
    if soft:
        augmented_pc, simulated_fog_pc, info_dict = P_R_fog_soft(p, augmented_pc, original_intensity, noise, gain,
                                                                 noise_variant)

    return augmented_pc, simulated_fog_pc, info_dict
