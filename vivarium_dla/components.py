import numpy as np, pandas as pd, matplotlib.pyplot as plt, sklearn.neighbors
import hashlib

from vivarium import Component

class DLA(Component):
    name = 'DLA'
    configuration_defaults = {
        'dla': {
            'stickiness': 0.9,
            'initial_position_radius': 50,
            'step_radius_rate': 5,
            'near_radius': 1,
            'n_start_frozen':3,
            'start_radius':1,
            'growth_rate':.4,  # percent per day
            'growth_stop_time': '2020-07-01'
        }
    }

    @property
    def columns_created(self):
        return ['x', 'y', 'z', 'frozen']
    
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
        
        
    def on_initialize_simulants(self, simulant_data):
        """Start new simulants at random location, but 
        make sure there is something frozen at the origin
        """
        pop = pd.DataFrame(index=simulant_data.index)
        pop['x'] = self.np_random.normal(size=len(simulant_data.index),
                                         scale=self.config.initial_position_radius)
        pop['y'] = self.np_random.normal(size=len(simulant_data.index),
                                         scale=self.config.initial_position_radius)
        pop['z'] = self.np_random.uniform(size=(len(simulant_data.index)), low=0, high=self.config.initial_position_radius/50)
        pop['frozen'] = np.nan

        # freeze first simulants in the batch
        for i in range(self.config.n_start_frozen):
            pop.iloc[i, :] = [self.config.initial_position_radius + self.config.start_radius * np.sin(2*np.pi*i/self.config.n_start_frozen),
                              self.config.start_radius * np.cos(2*np.pi*i/self.config.n_start_frozen),
                              2,
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
        pop.z += np.where(pop.frozen.isnull(), self.np_random.normal(size=len(pop.index),
                                                             scale=self.step_radius/50), 0)
        
        
        # freeze
        to_maybe_freeze = self.near_frozen(pop)  # TODO: make it clearer that this series includes the index of the node that this node froze to
        to_freeze =  (self.freeze_randomness.get_draw(to_maybe_freeze.index)
                      < self.config.stickiness)
        freeze_parent_index = to_maybe_freeze[to_freeze == True]
        pop.loc[freeze_parent_index.index, 'frozen'] = freeze_parent_index

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
            return pd.Series(dtype='float64')

        frozen = pop[~pop.frozen.isnull()].loc[:, ['x', 'y', 'z']]
        X = frozen.values
        
        tree = sklearn.neighbors.KDTree(X, leaf_size=2)
        
        num_near = tree.query_radius(not_frozen.values, r=self.config.near_radius, count_only=True)
        to_freeze = not_frozen[(num_near > 0)].index
        if len(to_freeze) == 0:
            return pd.Series(dtype='float64')
        index_near = tree.query_radius(not_frozen.loc[to_freeze].values, r=self.config.near_radius, count_only=False)
        
        return pd.Series(map(lambda x:frozen.index[x[0]], # HACK: get the index of the first frozen node close to this one
                             index_near), index=to_freeze)

class SaveImage(Component):
    name = 'SaveImage'
    configuration_defaults = {
        'dla': {
            'dname': '/tmp/',
        }
    }

    @property
    def columns_required(self):
        return ['x', 'y', 'z', 'frozen']

    def setup(self, builder):
        self.config = builder.configuration.dla
        self.step_size = builder.configuration.time.step_size
        self.pop_size = builder.configuration.population.population_size
        
        self.randomness = builder.randomness.get_stream('save_image')
        self.seed = self.randomness._key() # from https://github.com/ihmeuw/vivarium/blob/95ac55e4f5eb7c098d99fe073b35b73127e7ed0d/src/vivarium/framework/randomness/stream.py#L66

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
            plt.plot(xx, yy, '-', alpha=.85, linewidth=1, color='C0')
            
        
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
                    + f'n_frozen {pop.frozen.isnull().sum():,.0f}\n'
                    , ha='left', va='top')
        import datetime
        fname = (f'{self.config.dname}/{datetime.datetime.today().strftime("%Y%m%d")}-{self.config.stickiness}-'
                 + f'{self.config.initial_position_radius}-{self.config.step_radius_rate}-'
                 + f'{self.config.near_radius}-{self.config.growth_rate}-{self.config.growth_stop_time}-'
                 + f'{self.pop_size}-{self.step_size}-{self.seed[-5:]}.png')
        plt.savefig(fname)
        print(f'Visual results saved as {fname}')

        pop.to_csv(fname.replace('.png', '.csv.bz2'))

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


class BoundingBox(Component):
    name = 'BoundingBox'
    configuration_defaults = {
        'dla': {
            'bounding_box_radius': 100,
        }
    }

    @property
    def columns_required(self):
        return ['x', 'y']
    
    def setup(self, builder):
        self.config = builder.configuration.dla
        
    def on_time_step_prepare(self, event):
        pop = self.population_view.get(event.index)
        
        # wrap all points into the bounding box
        pop.x = np.clip(pop.x, -2*self.config.bounding_box_radius,
                        2*self.config.bounding_box_radius)
        pop.y = np.clip(pop.y, -2*self.config.bounding_box_radius,
                        2*self.config.bounding_box_radius)
        self.population_view.update(pop)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sklearn.neighbors
from vivarium import Component

class SaveFlipBook(Component):
    name = 'SaveFlipBook'
    configuration_defaults = {
        'dla': {
            'dname': '/tmp/',
            'figsize': (10, 10),
        }
    }

    @property
    def columns_required(self):
        return ['x', 'y', 'z', 'frozen']

    def setup(self, builder):
        self.config = builder.configuration.dla
        self.step_size = builder.configuration.time.step_size
        self.pop_size = builder.configuration.population.population_size
        
        self.randomness = builder.randomness.get_stream('save_flipbook')
        self.seed = self.randomness._key()
        
        # Initialize PDF writer
        import datetime
        self.fname = (f'{self.config.dname}/{datetime.datetime.today().strftime("%Y%m%d")}-flipbook-'
                     + f'{self.config.stickiness}-{self.config.initial_position_radius}-'
                     + f'{self.config.step_radius_rate}-{self.config.near_radius}-'
                     + f'{self.config.growth_rate}-{self.config.growth_stop_time}-'
                     + f'{self.pop_size}-{self.step_size}-{self.seed[-5:]}.pdf')
        self.pdf = PdfPages(self.fname)
        
        # Track frozen count to detect new freezes
        self.last_frozen_count = self.config.n_start_frozen
        
    def on_time_step(self, event):
        pop = self.population_view.get(event.index)
        current_frozen_count = (~pop.frozen.isnull()).sum()
        
        # Only save a new page if we have new frozen nodes
        if current_frozen_count > self.last_frozen_count:
            self.save_current_state(pop, event.time)
            self.last_frozen_count = current_frozen_count

    def save_current_state(self, pop, current_time):
        plt.figure(figsize=self.config.figsize)

        frozen = pop[~pop.frozen.isnull()].loc[:, ['x', 'y', 'z']]
        plt.plot(pop.iloc[:self.config.n_start_frozen].x, 
                pop.iloc[:self.config.n_start_frozen].y, 'o')
        
        # Draw connections between frozen particles
        mean_frozen_z = frozen.z.mean()
        for i in pop[~pop.frozen.isnull()].index:
            j = pop.loc[i, 'frozen']
            xx = [pop.x[i], pop.x[j]]
            yy = [pop.y[i], pop.y[j]]
            color = 'b' if pop.z[i] > mean_frozen_z else 'r'
            plt.plot(xx, yy, '-', alpha=.85, linewidth=1, color='C0')
        
        # Plot all particles
        plt.plot(pop.x, pop.y, 'k,')
        
        # Set bounds
        bnds = plt.axis()
        max_bnd = np.max(bnds)
        plt.axis(xmin=-max_bnd, xmax=max_bnd, 
                ymin=-max_bnd, ymax=max_bnd)
        
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Add metadata text
        plt.figtext(0, 1, 
                   f'Time: {current_time}\n'
                   f'Frozen particles: {self.last_frozen_count:,.0f}\n'
                   f'stickiness {self.config.stickiness}; '
                   f'initial_position_radius {self.config.initial_position_radius}; '
                   f'step_radius_rate {self.config.step_radius_rate}; '
                   f'near_radius {self.config.near_radius}',
                   ha='left', va='top')
        
        # Save to PDF and close the figure
        self.pdf.savefig()
        plt.close()
        
    def on_simulation_end(self, event):
        """Close the PDF file when simulation ends"""
        self.pdf.close()
        print(f'Flipbook saved as {self.fname}')