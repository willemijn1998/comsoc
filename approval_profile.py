# %%
import numpy as np
import pickle

from numpy import random
from pylab import rcParams
from collections import Counter
from datetime import date
# import matplotlib.pyplot as plt 
# %matplotlib inline
from matplotlib import pyplot as plt

def normal(loc=0, scale=1, size=None, **kwargs):
    return random.normal(loc, scale, size)

def uniform(low=0.0, high=1.0, size=None, **kwargs):
    return random.uniform(low, high, size)

class Cluster():
    def __init__(self, ballots):
        self._ballots = ballots
        self._mean = np.mean(ballots, 0)
        self.n_projects = len(ballots[0])


    @property
    def ballots(self):
        return self._ballots


    @property
    def cluster_ballot(self):
        return self._mean


    def __iter__(self):
        return iter(self._ballots)


    def __repr__(self):
        return str(self._mean)


    def __list__(self):
        return list(self._ballots)

    def __len__(self):
        return len(self._ballots)

    def statistics(self):
        rcParams['figure.figsize'] = 7, 4
        rcParams['figure.dpi'] = 75
        rcParams.update({'font.size': 10})
        print(f"Probability that a voter votes the same for one project compared to the cluster ballot = {1-np.mean(np.abs(self._ballots-self._mean))}")
        x,y = np.unique(self._ballots, return_counts=True, axis=0)
        print(f"Number of unique ballets in this cluster = {len(x)}")
        print(f"Number of voters that vote the cluster ballot = {max(y)}")
        plt.plot(*np.unique(np.abs(self._ballots - self._mean).sum(1), return_counts=True), '-o')
        plt.xlabel("difference between voter ballot and cluster ballot")
        plt.ylabel("number of voters")
        plt.show()
        plt.plot(self._mean*100, 'o',label="cluster ballot")
        plt.plot(self._ballots.mean(0)*100,'x',label="voters")
        plt.legend()
        plt.xlim(0,self.n_projects)
        plt.xlabel("Projects")
        plt.ylabel("Change of approving a project(%)")
        plt.show()


class Cluster_Generator():
    def __init__(self,
                 voters_per_cluster,
                 votes_per_project,
                 cluster_independence=3,
                 spread_of_approvals=1.5,
                 adcavpd=0.0,
                 sdcavpd=0.1,
                 noise=0.01,
                 **kwargs):
        self.voters_per_cluster = np.array(sorted(voters_per_cluster, reverse=False))
        self.votes_per_project = np.array(votes_per_project)
        self.cluster_independence = cluster_independence
        self.spread_of_approvals = spread_of_approvals
        self.avg_diff_cluster_all_voters_project_dist = adcavpd
        self.std_diff_cluster_all_voters_project_dist = sdcavpd
        self.noise = noise

        self.n_projects = len(votes_per_project)
        self.max_voters_per_cluster = max(voters_per_cluster)
        self.n_clusters = len(self.voters_per_cluster)
        self.rng = random.default_rng()


    @property
    def size_none_empty_projects(self):
        return len(self.none_empty_projects)


    @property
    def none_empty_projects(self):
        return np.where(self.votes_per_project != 0)[0]


    @property
    def votes_per_project_dist(self):
        return self.votes_per_project / sum(self.votes_per_project)


    @property
    def avg_approvals_per_voter(self):
        return sum(self.votes_per_project) / sum(self.voters_per_cluster)


    def convert_ballot_to_one_hot(self, ballots):
        new_ballots = np.zeros((len(ballots), self.n_projects))
        for i, vote in enumerate(ballots):
            new_ballots[i,vote] = 1
        return new_ballots


    def make_clusters(self):
        clusters = []
        for cluster_size in self.voters_per_cluster:
            npic = self.n_projects_in_cluster(cluster_size)
            ps = self.generate_project_set(npic)
            apv = self.approvals_per_voter(npic, cluster_size)
            cpd = self.gen_cluster_project_dist(ps)
            cpdn,psn = self.gen_cluster_project_dist_noise(cpd, ps)
            ballots = self.gen_cluster_ballot(psn, apv, cpdn)

            for proj, votes in Counter([y for x in ballots for y in x]).items():
                self.votes_per_project[proj] = np.maximum(self.votes_per_project[proj] - votes, 0)
            self.voters_per_cluster = np.delete(self.voters_per_cluster, 0)

            ballots = self.convert_ballot_to_one_hot(ballots)
            clusters.append(Cluster(ballots))

        return clusters


    def n_projects_in_cluster(self, cluster_size):
        # the lower cluster_independence is the more dependent the number of projects are in therms of clustersize.
        return int(np.clip(random.normal(self.size_none_empty_projects * (cluster_size / self.max_voters_per_cluster), self.cluster_independence, 1), 1, self.size_none_empty_projects))


    def generate_project_set(self, n_projects_in_cluster):
        return self.rng.choice(self.n_projects, n_projects_in_cluster, p=self.votes_per_project_dist, replace=False, shuffle=False)


    def approvals_per_voter(self, n_projects_in_cluster, cluster_size):
        # spread_of_approvals determines the spread of approvals per voter within a cluster.
        return np.clip(random.normal(self.avg_approvals_per_voter + 1, self.spread_of_approvals, cluster_size), 1, max(n_projects_in_cluster, 2)).astype(int)


    def gen_cluster_project_dist(self, project_set):
        # distribution for all projects (small values for non-cluster-projects)
        new_dist = np.maximum(self.votes_per_project_dist[project_set] + random.normal(self.avg_diff_cluster_all_voters_project_dist, self.std_diff_cluster_all_voters_project_dist, len(project_set)), self.noise)
        return new_dist / sum(new_dist)


    def gen_cluster_project_dist_noise(self, cluster_project_dist, project_set):
        noise_projects = list(set(range(self.n_projects)) - set(project_set))
        cpdn = np.array(list(cluster_project_dist) + list(np.clip(random.gamma(self.noise, 4, len(noise_projects)), 0, 3*self.noise)))
        return cpdn / sum(cpdn), list(project_set) + noise_projects


    def gen_cluster_ballot(self, project_set, approvals_per_voter, cluster_project_dist):
        return [self.rng.choice(project_set, apv, p=cluster_project_dist, replace=False, shuffle=False) for apv in approvals_per_voter]


    def __call__(self):
        return self.make_clusters()


class Profile():
    def __init__(self, filename):
        self._metadata = {}
        self._projects = {}
        self._votes = {}

        # cant pickle file object
        with open(filename, "r", encoding="utf8") as f:
            self.__read_lines(f)

        self.__convert_projects()
        self.__convert_votes()


    def __repr__(self):
        return str(self._metadata)


    @property
    def ballots(self):
        return self._ballots


    @property
    def approvals(self):
        return np.sum(self._ballots, 0)


    @property
    def budget(self):
        return float(self._metadata["budget"].replace(",", ".")) if isinstance(self._metadata["budget"], str) else self._metadata["budget"]


    @property
    def costs(self):
        return np.array([x[1] for x in sorted(self._projects.items(), key=lambda x: x[0])])


    def __convert_projects(self):
        self._projectid_to_index = {}
        tmp = {}
        for i, (proj_id, budget) in enumerate(self._projects.items()):
            self._projectid_to_index[proj_id] = i
            tmp[i] = budget
        self._projects = tmp


    def __convert_votes(self):
        self._votes = [np.array([self._projectid_to_index[x]]) if isinstance(x, int) else np.array([self._projectid_to_index[int(y)] for y in x.split(",")]) for x in self._votes.values()]
        self._ballots = np.zeros((self._metadata["num_votes"], self._metadata["num_projects"]))
        for i, vote in enumerate(self._votes):
            self._ballots[i,vote] = 1


    def __read_lines(self, f):
        _sections = {"META":self._metadata,
                     "PROJECTS":self._projects,
                     "VOTES":self._votes}
        _slices = {"key":"value",
                   "project_id":"cost",
                   "voter_id":"vote"}

        for line in f:
            line = line.strip()

            items = line.split(";")
            # find the right index for one of the properties (value, cost, vote)
            try:
                index = items.index(_slices[items[0]])
            except KeyError:
                pass
            else:
                continue

            # switch to a new dict when a new section is found
            try:
                _current = _sections[line]
            except KeyError:
                pass
            else:
                continue

            # read data
            try:
                try:
                    key = int(items[0])
                except ValueError:
                    key = items[0]
                _current[key] = int(items[index])
            except IndexError:
                pass
            except ValueError:
                _current[key] = items[index]


    def get_approval_percentage(self, projects):
        ballots_sliced = self._ballots[:,np.array(projects)]

        print(np.sum(np.sum(ballots_sliced, 1).astype(bool)))
        print(np.sum(ballots_sliced))

        return np.mean(np.sum(ballots_sliced, 1).astype(bool))


    def get_cost(self, projects):
        return sum(self.costs[np.array(projects)])


    def get_budget_percentage(self, projects):
        return self.get_cost(projects) / self.budget


    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

class Profile_Synthetic(Profile):
    def __init__(self,
                 votes_per_project=list(range(5000, 1500, -300)),
                 voters_per_cluster=list(range(1500, 100, -100)),
                 budget_distribution=normal,
                 loc = 5000,
                 scale = 2000,
                 **kwargs):
        self._clusters = Cluster_Generator(voters_per_cluster, votes_per_project, **kwargs)()
        self._projects = self.__project_generator(votes_per_project, budget_distribution=budget_distribution, loc=loc, scale=scale, **kwargs)
        self._metadata = self.__create_metadata(votes_per_project, budget_distribution=budget_distribution, loc=loc, scale=scale, **kwargs)
        self._ballots = np.array([ballot for cluster in self._clusters for ballot in cluster])


    @property
    def clusters(self):
        return self._clusters


    def __project_generator(self, votes_per_project, **kwargs):
        return {i:kwargs['budget_distribution'](**kwargs) for i in range(len(votes_per_project))}


    def __create_metadata(self, votes_per_project, **kwargs):
        if 'budget' in kwargs:
            budget = kwargs['budget']
        else:
            budget = kwargs['budget_distribution'](**kwargs) * len(votes_per_project) / 3
        budget = np.round(budget, 2)

        return {'description': 'Synthetic data',
                'country': 'Monty Python',
                'unit': 'Holy Grail',
                'subunit': 'Life of Brian',
                'instance': date.today().year,
                'district': 'The Meaning of Life',
                'num_projects': len(self._projects),
                'num_votes': len(self._clusters),
                'budget': budget,
                'vote_type': 'approval',
                'rule': 'random',
                'date_begin': date.today().strftime("%d.%m.%Y"),
                'date_end': date.today().strftime("%d.%m.%Y"),
                'min_length': 1,
                'max_sum_cost': budget,
                'language': 'python',
                'edition': 1}


if __name__ == '__main__':
    path = "data/profiles/"
    test = Profile("data/poland_warszawa_2019_ursynow.pb")

# %%
