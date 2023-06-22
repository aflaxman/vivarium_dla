import numpy as np, pandas as pd, matplotlib.pyplot as plt, sklearn.neighbors
import hashlib

class DLA:
    name = 'DLA'
    configuration_defaults = {
        'dla': {
            'stickiness': 0.9,
            'initial_position_radius': 100,
            'step_radius_rate': 10,
            'near_radius': 1,
            'n_start_frozen':5,
            'start_radius':1,
            'growth_rate':.4,  # percent per day
            'growth_stop_time': '2020-05-01'
        }
    }

    def setup(self, builder):
        self.config = builder.configuration.dla
        self.growth_factor = (1 + self.config.growth_rate / 100)**builder.configuration.time.step_size
        self.growth_stop_time = pd.Timestamp(self.config.growth_stop_time)
        self.step_radius = self.config.step_radius_rate * builder.configuration.time.step_size
        self.vivarium_randomness = builder.randomness.get_stream('dla')

        vivarium_seed = self.vivarium_randomness._key() # from https://github.com/ihmeuw/vivarium/blob/95ac55e4f5eb7c098d99fe073b35b73127e7ed0d/src/vivarium/framework/randomness/stream.py#L66
        np_seed = int(hashlib.sha1(vivarium_seed.encode('utf-8')).hexdigest(), 16) % (10 ** 8)  # from https://stackoverflow.com/questions/7585307/how-to-correct-typeerror-unicode-objects-must-be-encoded-before-hashing
        self.np_random = np.random.RandomState(seed=np_seed)
        self.freeze_randomness = builder.randomness.get_stream('dla_freeze')
        
        columns = ['x', 'y', 'z', 'frozen']
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
        pop['z'] = 0.0
        pop['frozen'] = np.nan

        # freeze first simulants in the batch
        for i in range(self.config.n_start_frozen):
            pop.iloc[i, :] = [self.config.start_radius * np.sin(2*np.pi*i/self.config.n_start_frozen),
                              self.config.start_radius * np.cos(2*np.pi*i/self.config.n_start_frozen),
                              0,
                              (i+1)%self.config.n_start_frozen]
        
        # update the population in the model
        self.population_view.update(pop)
        
    def on_time_step(self, event):
        pop = self.population_view.get(event.index)
        
        # move the not-frozen
        pop.x += np.where(pop.frozen.isnull(), self.np_random.normal(size=len(pop.index),
                                                             scale=self.step_radius), 0)
        pop.y += np.where(pop.frozen.isnull(), self.np_random.normal(size=len(pop.index),
                                                             scale=self.step_radius), 0)
        
        # freeze
        to_maybe_freeze = self.near_frozen(pop)
        to_freeze =  (self.freeze_randomness.get_draw(to_maybe_freeze.index)
                      < self.config.stickiness)
        to_freeze = to_maybe_freeze[to_freeze == True]
        pop.loc[to_freeze.index, 'frozen'] = to_freeze

        # print some info for debugging
        t = pd.concat([pop.loc[to_freeze],
                         pop.loc[to_freeze.index]
                        ])
        if len(t) > 0:
            print(t)
                         

        # grow
        # TODO: refactor this into a separate component
        if event.time < self.growth_stop_time:
            frozen = ~pop.frozen.isnull()
            pop.loc[frozen, ['x', 'y']] *= self.growth_factor

        # update the population in the model
        self.population_view.update(pop)
                
        
    def near_frozen(self, pop):
        not_frozen = pop[pop.frozen.isnull()].loc[:, ['x', 'y', 'z']]
        if len(not_frozen) == 0:
            return pd.Series()

        frozen = pop[~pop.frozen.isnull()].loc[:, ['x', 'y', 'z']]
        X = frozen.values
        
        tree = sklearn.neighbors.KDTree(X, leaf_size=2)
        
        num_near = tree.query_radius(not_frozen.values, r=self.config.near_radius, count_only=True)
        to_freeze = not_frozen[(num_near > 0)].index
        if len(to_freeze) == 0:
            return pd.Series()
        index_near = tree.query_radius(not_frozen.loc[to_freeze].values, r=self.config.near_radius, count_only=False)
        
        return pd.Series(map(lambda x:frozen.index[x[0]], # HACK: get the index of the first frozen node close to this one
                             index_near), index=to_freeze)

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

        columns = ['x', 'y', 'z', 'frozen']
        self.population_view = builder.population.get_view(columns)

        builder.event.register_listener('simulation_end', self.on_simulation_end)

    def on_simulation_end(self, event):
        pop = self.population_view.get(event.index)

        plt.figure(figsize=(20,20))

        frozen = pop[~pop.frozen.isnull()].loc[:, ['x', 'y', 'z']]
        #plt.plot(frozen.x, frozen.y, '.')
        plt.plot(pop.iloc[:self.config.n_start_frozen].x, pop.iloc[:self.config.n_start_frozen].y, 'o')
        
        tree = sklearn.neighbors.KDTree(frozen.values, leaf_size=2)

        nearest = tree.query_radius(frozen.values, r=self.config.near_radius, count_only=False)
        xx, yy = [], []
        #for i, N_i in enumerate(nearest):
        #    for j in N_i[:2]:
        #        xx += [frozen.iloc[i, 0], frozen.iloc[j, 0], np.nan]
        #        yy += [frozen.iloc[i, 1], frozen.iloc[j, 1], np.nan]
        #plt.plot(xx, yy, 'k-', alpha=.85, linewidth=2)

        mean_frozen_z = frozen.z.mean()
        for i in pop[~pop.frozen.isnull()].index:
            j = pop.loc[i, 'frozen']
            xx = [pop.x[i], pop.x[j]]
            yy = [pop.y[i], pop.y[j]]
            if pop.z[i] > mean_frozen_z:
                color = 'b'
            else:
                color = 'r'
            plt.plot(xx, yy, '-', alpha=.85, linewidth=1, color=color)
            
        
        bnds = plt.axis()
        max_bnd = np.max(bnds)
        plt.plot(pop.x, pop.y, 'k,')
        plt.axis(xmin=-max_bnd, xmax=max_bnd, ymin=-max_bnd, ymax=max_bnd)

        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.figtext(0, 1, f'\n    stickiness {self.config.stickiness}; '
                    + f'initial_position_radius {self.config.initial_position_radius}; '
                    + f'step_radius_rate {self.config.step_radius_rate}; near_radius {self.config.near_radius}; '
                    + f'bounding_box_radius {self.config.bounding_box_radius} seed {self.seed}\n'
                    + f'n_frozen {pop.frozen.sum():,.0f}\n'
                    , ha='left', va='top')
        import datetime
        fname = (f'{self.config.dname}/{datetime.datetime.today().strftime("%Y%m%d")}{self.config.stickiness}-'
                 + f'{self.config.initial_position_radius}-{self.config.step_radius_rate}-'
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
        pop.x = np.clip(pop.x, -2*self.config.bounding_box_radius,
                        2*self.config.bounding_box_radius)
        pop.y = np.clip(pop.y, -2*self.config.bounding_box_radius,
                        2*self.config.bounding_box_radius)
        self.population_view.update(pop)
