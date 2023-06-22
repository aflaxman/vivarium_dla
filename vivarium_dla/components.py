import numpy as np, pandas as pd, matplotlib.pyplot as plt, sklearn.neighbors
import hashlib

class DLA:
    name = 'DLA'
    configuration_defaults = {
        'dla': {
            'stickiness': 0.9,
            'initial_position_radius': 100,
            'step_radius': .1,
            'near_radius': 1,
        }
    }

    def setup(self, builder):
        self.config = builder.configuration.dla
        self.vivarium_randomness = builder.randomness.get_stream('dla')

        vivarium_seed = self.vivarium_randomness._key() # from https://github.com/ihmeuw/vivarium/blob/95ac55e4f5eb7c098d99fe073b35b73127e7ed0d/src/vivarium/framework/randomness/stream.py#L66
        np_seed = int(hashlib.sha1(vivarium_seed.encode('utf-8')).hexdigest(), 16) % (10 ** 8)  # from https://stackoverflow.com/questions/7585307/how-to-correct-typeerror-unicode-objects-must-be-encoded-before-hashing
        self.np_random = np.random.RandomState(seed=np_seed)
        self.freeze_randomness = builder.randomness.get_stream('dla_freeze')
        
        columns = ['x', 'y', 'frozen']
        self.population_view = builder.population.get_view(columns)
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=columns)
        
        builder.event.register_listener('time_step', self.on_time_step)
        
    def on_initialize_simulants(self, simulant_data):
        """Start new simulants at random location, but 
        make sure there is something frozen at the origin
        """
        pop = pd.DataFrame(index=simulant_data.index)
        pop['x'] = self.np_random.normal(size=len(simulant_data.index),
                                         scale=self.config.initial_position_radius)
        pop['y'] = self.np_random.normal(size=len(simulant_data.index),
                                         scale=self.config.initial_position_radius)

        pop['frozen'] = False

        # freeze first simulant in the batch 
        pop.iloc[0, :] = [0, 0, True]
        
        # update the population in the model
        self.population_view.update(pop)
        
    def on_time_step(self, event):
        pop = self.population_view.get(event.index)
        
        # move the not-frozen
        pop.x += np.where(~pop.frozen, self.np_random.normal(size=len(pop.index),
                                                             scale=self.config.step_radius), 0)
        pop.y += np.where(~pop.frozen, self.np_random.normal(size=len(pop.index),
                                                             scale=self.config.step_radius), 0)
        
        # freeze
        to_freeze = self.near_frozen(pop)
        pop.loc[to_freeze, 'frozen'] = (self.freeze_randomness.get_draw(to_freeze) < self.config.stickiness)
        self.population_view.update(pop)
        
    def near_frozen(self, pop):
        not_frozen = pop[~pop.frozen].loc[:, ['x', 'y']]
        if len(not_frozen) == 0:
            return []

        X = pop[pop.frozen].loc[:, ['x', 'y']].values
        tree = sklearn.neighbors.KDTree(X, leaf_size=2)
        
        num_near = tree.query_radius(not_frozen.values, r=self.config.near_radius, count_only=True)
        to_freeze = not_frozen[(num_near > 0)].index
        
        return to_freeze

class SaveImage:
    name = 'SaveImage'
    configuration_defaults = {
        'dla': {
            'dname': '/tmp/',
        }
    }

    def setup(self, builder):
        self.config = builder.configuration.dla
        self.randomness = builder.randomness.get_stream('save_image')
        self.seed = self.randomness._key() # from https://github.com/ihmeuw/vivarium/blob/95ac55e4f5eb7c098d99fe073b35b73127e7ed0d/src/vivarium/framework/randomness/stream.py#L66

        columns = ['x', 'y', 'frozen']
        self.population_view = builder.population.get_view(columns)

        builder.event.register_listener('simulation_end', self.on_simulation_end)

    def on_simulation_end(self, event):
        pop = self.population_view.get(event.index)

        plt.figure(figsize=(20,20))

        frozen = pop[pop.frozen].loc[:, ['x', 'y']]
        tree = sklearn.neighbors.KDTree(frozen.values, leaf_size=2)

        nearest = tree.query_radius(frozen.values, r=self.config.near_radius, count_only=False)
        xx, yy = [], []
        for i, N_i in enumerate(nearest):
            for j in N_i[:2]:
                xx += [frozen.iloc[i, 0], frozen.iloc[j, 0], np.nan]
                yy += [frozen.iloc[i, 1], frozen.iloc[j, 1], np.nan]
        plt.plot(xx, yy, 'k-', alpha=.85, linewidth=2)

        bnds = plt.axis()
        max_bnd = np.max(bnds)
        plt.axis(xmin=-max_bnd, xmax=max_bnd, ymin=-max_bnd, ymax=max_bnd)

        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.figtext(0, 1, f'\n    stickiness {self.config.stickiness}; '
                    + f'initial_position_radius {self.config.initial_position_radius}; '
                    + f'step_raduis {self.config.step_radius}; near_radius {self.config.near_radius}; '
                    + f'bounding_box_radius {self.config.bounding_box_radius} seed {self.seed}\n'
                    + f'n_frozen {pop.frozen.sum():,.0f}\n'
                    , ha='left', va='top')
        fname = (f'{self.config.dname}/{self.config.stickiness}-'
                    + f'{self.config.initial_position_radius}-{self.config.step_radius}-'
                    + f'{self.config.near_radius}-{self.config.bounding_box_radius}-{self.seed[-5:]}.png')
        plt.savefig(fname)
        print(f'Visual results saved as {fname}')

class ChaosMonkey:
    name = 'ChaosMonkey'
    configuration_defaults = {
        'chaos_monkey': {
            'probability': .5,
        }
    }

    def setup(self, builder):
        self.config = builder.configuration.chaos_monkey

        # intentionally _do not_ use common randomness here
        self.failure = np.random.uniform() < self.config.probability

        builder.event.register_listener('time_step', self.on_time_step)

    def on_time_step(self, event):
        if self.failure:
            # intentionally _do not_ use common randomness here
            if np.random.random() < .5:
                assert 0, 'chaos monkey strikes'


class BoundingBox:
    name = 'BoundingBox'
    configuration_defaults = {
        'dla': {
            'bounding_box_radius': 100,
        }
    }

    def setup(self, builder):
        self.config = builder.configuration.dla
        self.population_view = builder.population.get_view(['x', 'y'])
        builder.event.register_listener('time_step__prepare', self.on_time_step__prepare)
        
    def on_time_step__prepare(self, event):
        pop = self.population_view.get(event.index)
        
        # wrap all points into the bounding box
        pop.x = np.mod(pop.x + self.config.bounding_box_radius,
                       2*self.config.bounding_box_radius) - self.config.bounding_box_radius
        pop.y = np.mod(pop.y + self.config.bounding_box_radius,
                       2*self.config.bounding_box_radius) - self.config.bounding_box_radius
        self.population_view.update(pop)
