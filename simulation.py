from model_class import Agent, Model, plot_q, plot_q_paperlike, plot_regions_vs_L
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    model = Model(
        nx.Graph(), 
        feature_len=5, 
        grid_len=20, 
        traits_per_feature=6, 
        neighbourhood="standard")
    
    model.create_grid()
    #model.plot_grid(100_000)

    model = Model(
        nx.Graph(),
        feature_len=5,
        grid_len=20,
        traits_per_feature=5,
        neighbourhood="standard")
    model.create_grid()
    #model.plot_t(100_000)

    model = Model(
        nx.Graph(),
        feature_len=5,
        grid_len=20,
        traits_per_feature=5,
        neighbourhood="standard")

    #plot_q(2, 25, 5, 10, n=100_000)
    plot_q_paperlike(1, 25, 5, 10, trials=10, n=1_000_000)
    #plot_regions_vs_L([5,10,30,40,50],5,5, trials=5, n=10_000_000)
    
    
    """qs = range(2, 15)
    trials = 20
    means = []
    stds = []

    for q in qs:
        vals = []

        for _ in range(trials):
            model = Model(nx.Graph(), feature_len=5, grid_len=10, traits_per_feature=q)
            model.create_grid()
            model.run_simulation(50000, include_acc=False)
            vals.append(len(model.distinct_agents_traits()))

        means.append(np.mean(vals))
        stds.append(np.std(vals))

    plt.figure()

    plt.plot(qs, means)
    plt.fill_between(qs, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.3)

    plt.title("q vs number of cultures (mean ± std)")
    plt.show()

    L_values = [5, 10, 15, 20]
    trials = 10

    means = []
    stds = []

    for L in L_values:
        vals = []

        for _ in range(trials):
            model = Model(nx.Graph(), feature_len=5, grid_len=L, traits_per_feature=5)
            model.create_grid()
            model.run_simulation(50000, include_acc=False)
            vals.append(model.count_regions())

        means.append(np.mean(vals))
        stds.append(np.std(vals))

    plt.figure()

    plt.plot(L_values, means)
    plt.fill_between(L_values, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.3)

    plt.title("regions vs L (mean ± std)")
    plt.show()

    neighs = ["standard", "extended", "toroidal"]
    trials = 10

    means = []
    stds = []

    for ntype in neighs:
        vals = []

        for _ in range(trials):
            model = Model(nx.Graph(), feature_len=5, grid_len=10, traits_per_feature=5, neighbourhood=ntype)
            model.create_grid()
            model.run_simulation(50000, include_acc=False)
            vals.append(len(model.distinct_agents_traits()))

        means.append(np.mean(vals))
        stds.append(np.std(vals))

    plt.figure()

    plt.bar(neighs, means)
    plt.title("neighbourhood comparison (mean cultures)")
    plt.show()"""