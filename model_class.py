import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

'''
SPOSÓB DZIAŁANIA KLASY AGENT 

# 1. Tworzymy dwóch agentów z wektorem 5 cech, gdzie każda cecha przyjmuje wartość od 0 do 2
agent_a = Agent(traits_per_feature=3, feature_len=5)
agent_b = Agent(traits_per_feature=3, feature_len=5)

# Ręcznie nadpiszmy cechy, aby idealnie zobaczyć mechanizm
agent_a.features = np.array([1, 0, 2, 1, 0])
agent_b.features = np.array([1, 2, 2, 0, 0]) 
# Wspólne indeksy: 0 (wartość 1) oraz 4 (wartość 0) -> podobieństwo = 2/5 = 0.4

print(f"Agent A: {agent_a}")
print(f"Agent B: {agent_b}")

# 2. Obliczanie podobieństwa
prawdopodobienstwo = agent_a.is_similar(agent_b)
print(f"Prawdopodobieństwo interakcji (podobieństwo): {prawdopodobienstwo}")  # Wyjdzie 0.4

# 3. Zmiana cechy (Agent A przejmuje jedną cechę od Agenta B)
agent_a.change_feature(agent_b.features)
print(f"Agent A po interakcji: {agent_a} (zmienił się losowy, różniący się indeks!)")

'''
"""SPOSOB DZIALANIA KLASY MODEL
     Najważniejszym elementem tej klasy jest 
     self.grid - macierz w której przechowujemy elementy klasy Agent
        możemy ich znajdować po wspolrzednych self.grid[row][column]
        
     self.graph - graf do którego dodajemy Agentów 
        możemy znaleźć wszystkich sąsiadów korzystajac z self.get_neigbours(self.grid[row][col])

"""


class Agent:
    def __init__(self, traits_per_feature: int, feature_len: int):
        self.feature_len = feature_len
        self.traits_per_features = traits_per_feature
        self.features = np.random.randint(0, traits_per_feature, feature_len, dtype=int)

    def change_feature(self, adj_site_features: np.ndarray):
        mask = self.features != adj_site_features  # filtering different traits
        indices = np.flatnonzero(mask)
        if indices.size > 0:  # checking if is there anything to change, is the change possible
            index_to_change = np.random.choice(indices)  # choose only from
            self.features[index_to_change] = adj_site_features[index_to_change]

    def is_similar(self, second_agent: Agent):
        """ checks similarity between agents, example: [1,3,0], [1,0,3] -similarity 1/3
        :returns p - similarity (prawdopodobieństwo interakcji)"""
        mask = self.features == second_agent.features
        return np.sum(mask) / self.feature_len

    def __repr__(self):
        return "[" + " ".join(map(str, self.features)) + "]"


class Model:
    def __init__(self, graph: nx.Graph, feature_len: int, grid_len: int, traits_per_feature: int):
        self.graph = graph
        self.feature_len = feature_len
        self.grid_len = grid_len
        self.traits_per_feature = traits_per_feature
        self.grid = [[Agent(self.traits_per_feature, self.feature_len) for i in range(self.grid_len)] for j in
                     range(self.grid_len)]  # 2d matrix that help to locate agent

    def create_grid(self):
        """Fills up the graph from self.graph"""
        directions = [(1, 0), (0, 1)]  # predefined neighbourhood

        # Adding edges
        for row in range(self.grid_len):
            for col in range(self.grid_len):
                for dx, dy in directions:
                    x = row + dx
                    y = col + dy
                    if 0 <= x < self.grid_len and 0 <= y < self.grid_len:
                        self.graph.add_edge(self.grid[row][col], self.grid[x][y], color="black")

    def get_neighbours(self, agent: Agent):
        return list(self.graph.neighbors(agent))

    def has_converged(self):
        """Checks whether model converged into the stable state - nothing can be changed.
        In that case you can end simulation faster."""
        for u, v in self.graph.edges:
            similarity = u.is_similar(v)
            if 0 < similarity < 1:
                return False
        return True

    def simulation_trial(self):
        row, col = np.random.randint(0, self.grid_len), np.random.randint(0, self.grid_len)  # random 'active agent'
        neighbors = self.get_neighbours(self.grid[row][col])
        random_neighbor = random.choice(neighbors)  # random its neighbor
        similarity = self.grid[row][col].is_similar(random_neighbor)

        if np.random.uniform() < similarity:
            self.grid[row][col].change_feature(random_neighbor.features)

    def distinct_agents_traits(self):
        """Use it after simulation. Counts agents with distinct features """
        distinct = {}
        for i in self.grid:
            for j in i:
                temp = tuple(j.features.tolist())
                if temp not in distinct:
                    distinct[temp] = 1
                else:
                    distinct[temp] += 1
        return distinct

    def __repr__(self):
        return "\n".join(" ".join(map(str, row)) for row in self.grid)
    
    #################################################################################
    
    def avg_similarity(self):
        """Calculates the average similarity between each pair of neighbours in the graph.
        Returns:
            np.float(64): average similarity
        """
        similarities = []
        for u,v in self.graph.edges:
            similarities.append(u.is_similar(v))
        return np.mean(similarities)
    
    def run_simulation(self,n,acc=100):
        """Run the simulation of self for n steps or until it has converged if it happened faster.
        Args:
            n (int) : maximal number of iterations
            acc (int) : accuracy, number of iterations between each snapshot, default 100
        Returns:
            self (Model) : the self model after the simulation
            i (int) : number of iteration +1
            steps (int) : number of steps, taken every acc iterations
            cultures (int) : number of distinct cultures (Agents with distinct features), taken every acc iterations
            avg_similarities (np.float(64)) : value of average similarity between each pair of Agents, taken every acc iterations
        """
        steps=[]
        cultures=[]
        avg_similarities=[]
        i=1
        while i<=n:
            self.simulation_trial()
            if i % acc == 0:
                steps.append(i)
                cultures.append(len(self.distinct_agents_traits()))
                avg_similarities.append(self.avg_similarity())
            if i % 1000 == 0 and self.has_converged():
                break
            i+=1
        return self,i,steps,cultures,avg_similarities
    
    def plot_t(self,n,acc=100):
        """Create two plots after running the simulation: subplot 1 of cultures(time), subplot 2 of average_similarities(time)
        Args:
            n (int) : maximal number of iterations in the simulation
            acc (int) : accuracy, number of iterations between each snapshot in the simulation, defaults to 100
        """
        self,i,steps,cultures,avg_similarities = self.run_simulation(n,acc)
        x = np.array(steps)
        y1 = np.array(cultures)
        
        plt.figure(figsize=(10,4))
        plt.suptitle(f"Model Axelroda \n features = {self.feature_len}, traits per feature = {self.traits_per_feature}, siatka = {self.grid_len}x{self.grid_len}")
        
        plt.subplot(1,2,1)
        plt.plot(x,y1)
        plt.xlabel("kroki czasowe")
        plt.ylabel("liczba kultur")
        plt.title("Liczba kultur (różnych rodzajów agentów)")
        plt.grid(True)
        
        plt.subplot(1,2,2)
        plt.plot(steps,avg_similarities)
        plt.xlabel("kroki czasowe")
        plt.ylabel("średnie podobieństwo")
        plt.title("Średnie podobieństwo agentów w całym grafie")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def culture_matrix(self):
        """Create a matrix of cultures from self.
        Returns:
            matrix (np.ndarray): matrix of cultures
        """
        culture_ids = {}
        current_id= 0
        matrix = np.zeros((self.grid_len, self.grid_len))
        for i in range(self.grid_len):
            for j in range(self.grid_len):
                culture = tuple(self.grid[i][j].features)
                if culture not in culture_ids:
                    culture_ids[culture] = current_id
                    current_id += 1
                matrix[i, j] = culture_ids[culture]
        return matrix
        
    def plot_grid(self,n,acc=100):
        """ZRÓB TAK ŻEBY DZIAŁAŁO111111111111111
        dodaj funkcję culture_matrix zeby nie powtarzac kodu bez sensu plus musi być współny słownik kolorów do obu wizualizacji
        bo inaczej te same kolory oznaczają rózne kultury i troche to nie ma sensu, albo wgl usunąć, 
        wsm trochę ma sensu bo bedzie widać czy się skupiaja
        
        moze jakby było mniej wymiarów to kodować kolory wg podobienstwa, np jakby były 3 featury to kazdy mialby 
        jeden kanal znormalizowany rgb po q i imo by było widac na tych kolorach czy są kultury podbne do sb czy n
        ale nwm czt to jest potrzebne, szczegolnie ze trochę nwm co z wyższymi wymiarami, może zlockować walor na jakims poziomie
        przy mieszaniu kolorów (jak liczbowo mieszać kolory rgb bez zmiany jasności? nie wiem) i 
        jasność bd kolejnym wymiarem, zawsze pozostają wzorki xD 

        Args:
            n (int): number of iterations in a simulation
            acc (int, optional): accuracy of the simulation (go to -> run_simulation() for more info). Defaults to 100.
        """
        
        model_before = copy.deepcopy(self) # żeby nie uciekało
        matrix_bf = model_before.culture_matrix()
        
        model_after = self.run_simulation(n,acc)[0]
        matrix_af = model_after.culture_matrix
        
        all_cultures = matrix_bf
        for i,j in matrix_af:
            if matrix_af[i][j] not in all_cultures:
                all_cultures.append(matrix_af[i][j])
                
        colormap = {} ##to na 100% tak nie działa xddddd
        for iden in all_cultures(axis=0):
            colormap.update({iden:iden/256})

        plt.figure(figsize=(8,8))
        plt.suptitle("Wizualizacja kultur w modelu Axelroda")
        
        plt.subplot(1,2,1)
        plt.imshow(matrix_bf, cmap="tab20")
        plt.colorbar(label="Kultura")
        plt.title("Początkowy stan modelu")
        
        plt.subplot(1,2,2)
        plt.imshow(matrix_af, cmap="tab20")
        plt.colorbar(label="Kultura")
        plt.title("Końcowy stan modelu")
        
        plt.tight_layout()
        plt.show()
        
    
def plot_q(q0,qN,a,b,n=1000):
    """Create a plot of number of different cultures in a function of the number of traits per feature.
    Args:
        q0 (int): starting trait number
        qN (int): ending trait number
        a (int): number of features
        b (int): length of grid(denoted L in the paper)
        n (int, optional): number of iterations in the simulation. Defaults to 1000.
    """
    qs = range(q0, qN)
    final_cultures = []
    for q in qs:
        model = Model(nx.Graph(),feature_len=a,grid_len=b,traits_per_feature=q)
        model.create_grid()
        model.run_simulation(n)
        final_cultures.append(
            len(model.distinct_agents_traits())
        )
    plt.plot(qs, final_cultures, 'o-')
    plt.xlabel("q (traits per feature)")
    plt.ylabel("Liczba kultur (po zbieżności)")
    plt.title("Przejście fazowe (?) w modelu Axelroda")
    plt.grid()
    plt.show()


if __name__ == "__main__":

    model = Model(nx.Graph(), 6, 10, 2)

    model.create_grid()
    print(model.graph)

    # Example:  extracting neigbors and calculating similarity
    print(model.get_neighbours(model.grid[0][0]))
    choice = np.random.choice(model.get_neighbours(model.grid[0][0]))
    print(type(choice))
    print(model.grid[0][0], choice)
    print(model.grid[0][0].is_similar(choice))

    print("-----------------------")
    # whole ass simulation
    row = 0  # random 'active agent' - the one that may change
    col = 0
    neighbors = model.get_neighbours(model.grid[row][col])
    random_neighbor = np.random.choice(neighbors)  # random its neighbor
    print(random_neighbor, "neigbor")
    print(model.grid[row][col], "actve agent")
    similarity = model.grid[row][col].is_similar(random_neighbor)
    print(similarity)  # probaibility of change one of its features
    if np.random.uniform() < similarity:
        model.grid[row][col].change_feature(random_neighbor.features)
        print(random_neighbor, "neighbor")
        print(model.grid[row][col], "active agent")

    # Simulation example
    for i in range(100000):
        model.simulation_trial()
        if i % 1000 == 0 and model.has_converged():
            print(f"Model has converged in {i} steps")
            break
    print(model)
    print(model.distinct_agents_traits())
    
    print("############################")
    #after_sim,t = model.run_simulation(10)
    #print(after_sim,t)
    #model.plot()
    #after_sim.plot()
    model2 = Model(nx.Graph(), 7, 8, 10)
    model2.create_grid()
    #model2.plot_t(1000)
    #model.plot_grid(1000) to na razie nie działa
    plot_q(2,25,5,20)
