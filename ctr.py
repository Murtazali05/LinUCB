import numpy as np

from policy import linucb_policy


def ctr_simulator(K_arms, d, alpha, data_path):
    # Инициализируем policy
    linucb_policy_object = linucb_policy(K_arms=K_arms, d=d, alpha=alpha)

    # Instantiate trackers
    aligned_time_steps = 0
    cumulative_rewards = 0
    aligned_ctr = []
    unaligned_ctr = []  # for unaligned time steps

    # Open data
    with open(data_path, "r") as f:

        for line_data in f:

            # 1st column: Logged data arm.
            # Integer data type
            data_arm = int(line_data.split()[0])

            # 2nd column: Logged data reward for logged chosen arm
            # Float data type
            data_reward = float(line_data.split()[1])

            # 3rd columns onwards: 100 covariates. Keep in array of dimensions (100,) with float data type
            # Учитывая контекст т.е. covariates, находим ручку с
            covariate_string_list = line_data.split()[2:]
            data_x_array = np.array([float(covariate_elem) for covariate_elem in covariate_string_list])

            # Находим policy's chosen arm
            arm_index = linucb_policy_object.select_arm(data_x_array)

            # Проверяем если arm_index совпадает data_arm (т.е. такое же действие было выбрано)
            # Отметим, что data_arms index варьируется от 1 до 10 в то время как policy_arms варьируется от 0 до 9.
            if arm_index + 1 == data_arm:
                # Используем reward information для того, чтобы обновить выбранную ручку
                linucb_policy_object.linucb_arms[arm_index].reward_update(data_reward, data_x_array)

                # Вычисляем CTR
                aligned_time_steps += 1
                cumulative_rewards += data_reward
                aligned_ctr.append(cumulative_rewards / aligned_time_steps)

    return aligned_time_steps, cumulative_rewards, aligned_ctr, linucb_policy_object