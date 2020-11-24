import numpy as np
import matplotlib.pyplot as plt

class StatTracker:
  def __init__(self, herdsmen):
    self.herdsmen = herdsmen
    self.reset()

  def reset(self):
    self.total_animals = []
    self.mean_herdsize = []
    self.deaths = []
    self.dead_oldage = 0
    self.dead_disease = {}
    self.dead_health = 0
    self.health = []
    self.f_history = {}
    self.f_values = {}
    self.occupancy = []

  def record_occupancy(self, occ):
    self.occupancy.append(occ)

  def vaccinate_decision(self, disease, v):
    if disease in self.f_values:
      self.f_values[disease].append(v)
    else:
      self.f_values[disease] = [v]

  def oldage_death(self):
    self.dead_oldage += 1

  def health_death(self):
    self.dead_health += 1

  def disease_death(self, disease):
    if disease in self.dead_disease:
      self.dead_disease[disease] = self.dead_disease[disease]+1
    else:
      self.dead_disease[disease] = 1

  def step(self):
    herd_sizes = np.array([hman.herd.size() for hman in self.herdsmen.agents])

    h = []
    for herdsman in self.herdsmen.agents:
      for animal in herdsman.herd.animals:
        h.append(animal.health)
    self.health.append(np.average(np.array(h)))

    self.total_animals.append(np.sum(herd_sizes))
    self.mean_herdsize.append(np.average(herd_sizes))
    if 'rvf' in self.dead_disease:
      rvf_deaths = self.dead_disease['rvf']
    else:
      rvf_deaths = 0
    if 'cbpp' in self.dead_disease:
      cbpp_deaths = self.dead_disease['cbpp']
    else:
      cbpp_deaths = 0

    self.deaths.append((self.dead_oldage, self.dead_health, rvf_deaths, cbpp_deaths))
    self.dead_oldage = 0
    self.dead_health = 0
    self.dead_disease = {}

    for disease in self.f_values:
      frac = np.sum(np.array(self.f_values[disease])) / len(self.f_values[disease])
      if disease in self.f_history:
        self.f_history[disease].append(frac)
      else:
        self.f_history[disease] = [frac]

    self.f_values = {}

  def dump_plots(self):
    plt.plot(self.total_animals)
    axes = plt.gca()
    axes.set_ylim([0.0, max(self.total_animals)])
    plt.savefig('total-animals.png')
    plt.clf()

    plt.plot(self.mean_herdsize)
    axes = plt.gca()
    axes.set_ylim([0.0, max(self.mean_herdsize)])
    plt.savefig('mean-herdsize.png')
    plt.clf()

    plt.plot(self.health)
    axes = plt.gca()
    axes.set_ylim([0.0, max(self.health)])
    plt.savefig('mean-health.png')
    plt.clf()

    plt.plot(self.f_history)
    axes = plt.gca()
    axes.set_ylim([min(0,min(self.f_history)), max(self.f_history)])
    plt.savefig('vaccination_decisions.png')
    plt.clf()

    labels = ['old age','health','rvf','cbpp']
    for i in range(len(labels)):
      plt.plot([x[i] for x in self.deaths], label=labels[i])
    axes = plt.gca()
    axes.set_ylim([0.0, 20.0])
    plt.legend()
    plt.savefig('deaths.png')
    plt.clf()

    occ = self.occupancy[0]
    for i in range(1,len(self.occupancy)):
      occ = occ + self.occupancy[i]
    plt.imshow(occ, cmap='hot', interpolation='nearest')
    plt.savefig('heatmap.png')
    plt.clf()

    np.savetxt("heatmap.csv", occ, delimiter=",")

  def print_stats(self,show_labels):
    labels = ['old age','health','rvf','cbpp']
    totals = np.sum(np.array(self.deaths),axis=0)
    s = f'{totals[0]},{totals[1]},{totals[2]},{totals[3]}'
    herd_sizes = np.array([hman.herd.size() for hman in self.herdsmen.agents])
    s = f'{s},{herd_sizes.mean()},{herd_sizes.min()},{herd_sizes.max()}'
    ds = []
    for disease in self.f_history:
      d = np.array(self.f_history[disease])
      frac = d.mean()
      s = f'{s},{frac}'
      ds.append(disease)
    dlabels = ','.join(ds)
    if show_labels:
      print("d_oldage,d_health,d_rvf,d_cbpp,hsize_mean,hsize_min,hsize_max,"+dlabels)
    print(s)