import glob
import itertools
import logging
import os
import re

import click
import numpy as np
import pandas as pd

import pyspiel
from open_spiel.python import rl_agent, rl_environment
from open_spiel.python.policy import TabularPolicy, tabular_policy_from_policy


def tabular_policy_from_csv(game, filename):
    csv = pd.read_csv(filename, index_col=0)

    empty_tabular_policy = TabularPolicy(game)
    for state_index, state in enumerate(empty_tabular_policy.states):
        action_probabilities = {
                action: probability
                for action, probability in enumerate(csv.loc[state.history_str()])
                if probability > 0
            }
        infostate_policy = [
            action_probabilities.get(action, 0.)
            for action in range(game.num_distinct_actions())
        ]
        empty_tabular_policy.action_probability_array[
            state_index, :] = infostate_policy
    return empty_tabular_policy

def policy_to_csv(game, policy, filename):
    tabular_policy = tabular_policy_from_policy(game, policy)
    df = pd.DataFrame(
            data=tabular_policy.action_probability_array,
            index=[s.history_str() for s in tabular_policy.states])
    df.to_csv(filename)


def play_match(game, csv_policy_home, csv_policy_away):
    action_string = None

    agents = [
        tabular_policy_from_csv(game, csv_policy_home),
        tabular_policy_from_csv(game, csv_policy_away)
    ]

    env_configs = {"players": 2}
    env = rl_environment.Environment(game, **env_configs)
    num_actions = env.action_spec()["num_actions"]

    state = game.new_initial_state()

    # Print the initial state
    print(str(state))

    def sample_action(state, player_id):
        cur_legal_actions = state.legal_actions(player_id)
        # Remove illegal actions, re-normalize probs
        probs = np.zeros(num_actions)
        policy_probs = agents[player_id].action_probabilities(state, player_id=player_id)
        for action in cur_legal_actions:
            probs[action] = policy_probs[action]
        probs /= sum(probs)
        action = np.random.choice(len(probs), p=probs)
        return action

    while not state.is_terminal():
         # The state can be three different types: chance node,
        # simultaneous node, or decision node
        if state.is_chance_node():
            # Chance node: sample an outcome
            outcomes = state.chance_outcomes()
            num_actions = len(outcomes)
            print("Chance node, got " + str(num_actions) + " outcomes")
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            print("Sampled outcome: ",
                  state.action_to_string(state.current_player(), action))
            state.apply_action(action)

        elif state.is_simultaneous_node():
            # Simultaneous node: sample actions for all players.
            chosen_actions = [
                sample_action(state, pid)
                for pid in xrange(game.num_players())
                ]
            print("Chosen actions: ", [
              state.action_to_string(pid, action)
              for pid, action in enumerate(chosen_actions)
              ])
            state.apply_actions(chosen_actions)

        else:
            # Decision node: sample action for the single current player
            action = sample_action(state, state.current_player())
            action_string = state.action_to_string(state.current_player(), action)
            print("Player ", state.current_player(), ", randomly sampled action: ",
                  action_string)
            state.apply_action(action)

        print("New state: ", str(state))

    # Game is now done. Print utilities for each player
    returns = state.returns()
    for pid in range(game.num_players()):
      print("Utility for player {} is {}".format(pid, returns[pid]))
    return returns


def play_tournament(game, modeldir, rounds=100):
    results = []
    teams = set([
            re.search(r"(.+)_p\d\.csv", os.path.basename(f)).group(1)
            for f in glob.glob(modeldir + "/*.csv", recursive=True)])
    ranking = {}
    for team in teams:
        ranking[team] = 0
    for i in range(rounds):
        for (team1, team2) in list(itertools.combinations(teams, 2)):
            result = play_match(game, os.path.join(modeldir, f"{team1}_p1.csv"), os.path.join(modeldir, f"{team2}_p2.csv"))
            results.append({
                "team1": team1,
                "team2": team2,
                "score1": result[0],
                "score2": result[1]
                })
            if result[0] > result[1]:
                ranking[team1] += 3
            elif result[0] == result[1]:
                ranking[team1] += 1
                ranking[team2] += 1
            else:
                ranking[team2] += 3
            result = play_match(game, os.path.join(modeldir, f"{team2}_p1.csv"), os.path.join(modeldir, f"{team1}_p2.csv"))
            results.append({
                "team1": team2,
                "team2": team1,
                "score1": result[0],
                "score2": result[1]
                })
            if result[0] > result[1]:
                ranking[team2] += 3
            elif result[0] == result[1]:
                ranking[team1] += 1
                ranking[team2] += 1
            else:
                ranking[team1] += 3
    return ranking, results


@click.command()
@click.argument('game', type=str)
@click.argument('modeldir', type=click.Path(exists=True))
@click.argument('outputdir', type=click.Path(exists=True))
@click.option('--rounds', default=20, help='Number of rounds to play.')
def cli(game, modeldir, rounds):
    """Play a round robin tournament"""
    game = pyspiel.load_game(game)
    ranking, results = play_tournament(game, modeldir, rounds)
    pd.DataFrame(ranking).to_csv(os.path.join(outputdir, 'ranking.csv'))
    pd.DataFrame(results).to_csv(os.path.join(outputdir, 'results.csv'))

if __name__ == '__main__':
    cli()
