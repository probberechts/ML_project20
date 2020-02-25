# ML Project 2020

This repository contains the code to setup the final evaluation of the course "[Machine Learning: Project](https://onderwijsaanbod.kuleuven.be/syllabi/e/H0T25AE.htm)" (KU Leuven, Faculty of engineering, Department of Computer Science, [DTAI research group](https://dtai.cs.kuleuven.be)).

## Installation

The easiest setup is to simply copy the module `tournaments.py` to your own code repository. You will need `Python>=3.6`, `openspiel`, `pandas`, `numpy` and `click`.

## Usage

### Exporting an agent to a CSV file
You will have to submit your agent's policy (both as 1st and 2nd player) as a CSV file. These CSV files should follow the `<teamname>_p<pid>.csv` naming convention. For example, if you are team `robberechts_meert`, you should deliver two csv files: `robberechts_meert_p1.csv` and `robberechts_meert_p2.csv`. You can use the code below to create these CSV files:

```python
from tournament import policy_to_csv

game = pyspiel.load_game("lecuc_poker")
policy = <your policy for agent 1>
policy_to_csv(game, policy, '/models/agent_p1.csv')
```

Note that you can use `PolicyFromCallable` to create a policy from any function `f(state) : action`.

```python
def _random_policy(state):
  actions = state.legal_actions()
  return np.choice(actions)

game = pyspiel.load_game("lecuc_poker")
callable_policy = policy.PolicyFromCallable(game, _random_policy)
```

### Playing a tournament

For the final evaluation of your agent we will play a round robin (pairwise) tournament, in which each agent will play many games (pairwise) against all other agents. 

```sh
python tournament.py 'leduc_poker' <modeldir> <outputdir> --rounds 20
```

This script will generate two files `ranking.csv` (containing the final rankings) and `results.csv` (containing the result of each individual game). On these outcomes we will do a game-theoretic analysis to determine the winners.


