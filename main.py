from ctr import ctr_simulator
import matplotlib.pyplot as plt


if __name__ == '__main__':
    alpha_input = 1.5
    data_path = "data/dataset.txt"
    aligned_time_steps, cum_rewards, aligned_ctr, policy = ctr_simulator(K_arms=10, d=100, alpha=alpha_input,
                                                                         data_path=data_path)
    print("Total reward = ", cum_rewards)
    print("Aligned time steps:", aligned_time_steps)

    plt.plot([i for i in range(0, len(aligned_ctr))], aligned_ctr)
    plt.xlabel("time step(s)")
    plt.ylabel("ctr")
    plt.show()


