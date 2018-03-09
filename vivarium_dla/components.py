import numpy as np, pandas as pd, sklearn.neighbors

class DLA:
    configuration_defaults = {
        'lda': {
            'stickiness': 0.9,
            'initial_position_sd': 100,
            'step_sd': .1,
            'near_radius': 1,
        }
    }

    def setup(self, builder):
        self.config = builder.configuration.lda
        
        columns = ['x', 'y', 'frozen']
        self.population_view = builder.population.get_view(columns)
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=columns)
        
        builder.event.register_listener('time_step', self.on_time_step)
        
    def on_initialize_simulants(self, simulant_data):
        """Start new simulants at random location, but 
        make sure there is something frozen at the origin
        """
        pop = pd.DataFrame(index=simulant_data.index)
        
        pop['x'] = np.random.normal(scale=self.config.initial_position_sd, size=len(pop))
        pop['y'] = np.random.normal(scale=self.config.initial_position_sd, size=len(pop))

        pop['frozen'] = False

        # freeze first simulant in the batch 
        pop.iloc[0, :] = [0, 0, True]
        
        # update the population in the model
        self.population_view.update(pop)
        
    def on_time_step(self, event):
        pop = self.population_view.get(event.index)
        
        # move the not-frozen
        pop.x += np.where(~pop.frozen, np.random.normal(scale=self.config.step_sd, size=len(pop)), 0)
        pop.y += np.where(~pop.frozen, np.random.normal(scale=self.config.step_sd, size=len(pop)), 0)
        
        # freeze
        to_freeze = self.near_frozen(pop)
        pop.loc[to_freeze, 'frozen'] = (np.random.normal(size=len(to_freeze)) < self.config.stickiness)
        self.population_view.update(pop)
        
    def near_frozen(self, pop):
        X = pop[pop.frozen].loc[:, ['x', 'y']].values
        tree = sklearn.neighbors.KDTree(X, leaf_size=2)
        
        not_frozen = pop[~pop.frozen].loc[:, ['x', 'y']]
        num_near = tree.query_radius(not_frozen.values, r=self.config.near_radius, count_only=True)
        to_freeze = not_frozen[(num_near > 0)].index
        
        return to_freeze
