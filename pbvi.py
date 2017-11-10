# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

class Pomdp:
  def __init__(self, states=None, observations=None, actions=None, transitions=None, observe=None, reward=None, discount=None):
    self.states = states
    self.observations = observations
    self.actions = actions
    self.transitions = transitions
    self.observe = observe
    self.reward = reward
    self.discount = discount

class Solver:
  def __init__(self, pomdp, beliefs):
    self.pomdp = pomdp
    self.beliefs = beliefs
    self.values = [0]

  def backup(self):
    action_belief_terms = dict()
    for a in self.pomdp.actions:
      observation_terms = dict()
      for o in self.pomdp.observations:
        observation_terms[o] = [self.pomdp.discount * np.dot(self.pomdp.transitions[:,a,:] * self.pomdp.observe[a,:,o], v) for v in self.values]
      for i, b in enumerate(self.beliefs):
        action_belief_terms[(a, i)] = self.pomdp.reward[:,a]
        for o in self.pomdp.observations:
          action_belief_terms[(a, i)] += max([np.dot(w, b) for w in observation_terms[o]])
    new_values = []
    for i, b in enumerate(self.beliefs):
      f = lambda x : np.dot(x, b)
      new_values.append(max([action_belief_terms[(a, i)] for a in self.pomdp.actions], key=f))
    self.values = new_values

if __name__ == "__main__":
  pomdp = Pomdp(states=np.array([0, 1]),
                observations=np.array([0, 1]),
                actions=np.array([0, 1]),
                transitions=np.array([[[.1, .2], [.3, .4]],
                                      [[.9, .8], [.7, .6]]]),
                observe=np.array([[[.7, .4], [.1, .2]],
                                  [[.3, .6], [.9, .8]]]),
                reward=np.array([[1, 0], [.2, .8]]),
                discount=0.9)
  solver = Solver(pomdp, np.array([[1, 0], [.5, .5], [0, 1]]))
  print solver.values
  solver.backup()
  print solver.values
