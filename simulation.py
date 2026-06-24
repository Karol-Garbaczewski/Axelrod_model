from model_class import Agent, Model, plot_q, plot_q_paperlike, plot_regions_vs_L, plot_t_compare_neighbourhoods
import networkx as nx

if __name__ == "__main__":

    model = Model(
        nx.Graph(), 
        feature_len=3, 
        grid_len=18, 
        traits_per_feature=6, 
        neighbourhood="standard")
    
    #model.create_grid()
    #model.plot_grid(1_300_000, acc=50)

    model = Model(
        nx.Graph(),
        feature_len=3,
        grid_len=18,
        traits_per_feature=6,
        neighbourhood="standard")
    
    #model.create_grid()
    #model.plot_t(1_000_000, acc=1000)

    model = Model(
        nx.Graph(),
        feature_len=3,
        grid_len=18,
        traits_per_feature=6,
        neighbourhood="standard")

    #plot_q(2, 20, 5, 10, n=30_000, trials=4)
    #plot_q_paperlike(1, 20, 5, 10, trials=15, n=60_000)
    plot_regions_vs_L([10,30,50,100],5,15, trials=1, n=500_000_000)
    #plot_t_compare_neighbourhoods(200_000, 370, 6, 30, 12)