from model_class import Agent, Model, plot_q, plot_q_paperlike, plot_regions_vs_L
import networkx as nx

if __name__ == "__main__":

    model = Model(
        nx.Graph(), 
        feature_len=3, 
        grid_len=18, 
        traits_per_feature=6, 
        neighbourhood="standard")
    
    model.create_grid()
    model.plot_grid(1_300_000, acc=50)

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
    #plot_q_paperlike(1, 25, 5, 10, trials=10, n=1_000_000)
    #plot_regions_vs_L([5,10,30,40,50],5,5, trials=5, n=10_000_000)