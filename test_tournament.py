import os
import pandas as pd
import pytest

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import get_all_states
from tournament import policy_to_csv, tabular_policy_from_csv, play_match, play_tournament

@pytest.fixture
def csv_policy(tmpdir):
    # Setup game and policy
    game = pyspiel.load_game("kuhn_poker")
    tabular_policy = policy.TabularPolicy(game)
    # Save policy as CSV
    output = os.path.join(tmpdir, 'policy.csv')
    policy_to_csv(game, tabular_policy, output)
    return output

def test_tabular_policy_to_csv(tmpdir):
    # Setup game and policy
    game = pyspiel.load_game("kuhn_poker")
    tabular_policy = policy.TabularPolicy(game)
    # Save policy as CSV
    output = os.path.join(tmpdir, 'policy.csv')
    policy_to_csv(game, tabular_policy, output)
    assert list(tmpdir.listdir()) == [output]
    # Check created CSV
    csv = pd.read_csv(output, index_col=0)
    # Get all states in the game at which players have to make decisions.
    states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=False)
    assert set(csv.index.values) <= set(states.keys())
    assert len(csv.columns) == game.num_distinct_actions()

def test_callable_policy_to_csv(tmpdir):
    def _uniform_policy(state):
        actions = state.legal_actions()
        p = 1.0 / len(actions)
        return [(a, p) for a in actions]

    # Setup game and policy
    game = pyspiel.load_game("kuhn_poker")
    callable_policy = policy.PolicyFromCallable(game, _uniform_policy)
    # Save policy as CSV
    output = os.path.join(tmpdir, 'policy.csv')
    policy_to_csv(game, callable_policy, output)
    assert list(tmpdir.listdir()) == [output]
    # Check created CSV
    csv = pd.read_csv(output, index_col=0)
    # Get all states in the game at which players have to make decisions.
    states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=False)
    assert set(csv.index.values) <= set(states.keys())

def test_tabular_policy_from_csv(tmpdir):
    game = pyspiel.load_game("kuhn_poker")
    output = os.path.join(tmpdir, 'policy.csv')
    tabular_policy = policy.TabularPolicy(game)
    # Save policy as CSV
    output = os.path.join(tmpdir, 'policy.csv')
    policy_to_csv(game, tabular_policy, output)
    tabular_policy_from_csv(game, output)


def test_play_match(csv_policy):
    game = pyspiel.load_game("kuhn_poker")
    result = play_match(game, csv_policy, csv_policy)
    assert len(result) == 2


def test_play_tournament(tmpdir):
    game = pyspiel.load_game("kuhn_poker")
    for team in ["python", "ruby", "java"]:
        for player in ["p1", "p2"]:
            tabular_policy = policy.TabularPolicy(game)
            # Save policy as CSV
            output = os.path.join(tmpdir, f'{team}_{player}.csv')
            policy_to_csv(game, tabular_policy, output)
    ranking, results = play_tournament(game, str(tmpdir))
    assert len(list(ranking.keys())) == 3
    assert len(results) == 3*2*2 


