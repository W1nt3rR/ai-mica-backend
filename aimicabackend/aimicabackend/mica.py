import json
import os
import time
from multiprocessing import Pool, Manager
from typing import List
from aimicabackend.types import TGameState, TMapData, TPoint, TMills, TPlayer, TDifficulty

memo = {}

def worker(args):
    child_state, child_move_from, child_move_to, depth, difficulty, map_data, mills, start_time, timeout = args
    eval = minimax(child_state, child_move_from, child_move_to, depth, difficulty, float('-inf'), float('inf'), False, map_data, mills, start_time, timeout)
    return child_state, eval

def get_best_move(game_state: TGameState, depth: int, map_name: str, difficulty: TDifficulty, timeout: int) -> TGameState:
    start_time = time.time()
    map_data = load_map_file(map_name)
    mills = precalculate_mills(map_data)

    with Pool() as pool:
        results = pool.map(worker, [(child_state, child_move_from, child_move_to, depth, difficulty, map_data, mills, start_time, timeout) for child_state, child_move_from, child_move_to in get_possible_moves(game_state, map_data, mills)])

    best_move, max_eval = max(results, key=lambda x: x[1], default=(None, float('-inf')))

    return {
        'gameState': best_move,
        'eval': max_eval
    }

def minimax(game_state: TGameState, point_from: TPoint, point_to: TPoint, depth: int, difficulty: TDifficulty, alpha: float, beta: float, maximizing_player: bool, map_data: TMapData, mills: TMills, start_time: float, timeout: float) -> float:
    # Convert the game state to a string
    game_state_str = json.dumps(game_state, sort_keys=True)
    point_from_str = json.dumps(point_from, sort_keys=True)
    point_to_str = json.dumps(point_to, sort_keys=True)
    memo_key = (game_state_str, point_from_str, point_to_str)

    # Check if the game state has already been evaluated
    if memo_key in memo:
        score, score_depth, difficulty = memo[memo_key]
        if score_depth == depth and difficulty == difficulty:
            return score

    if depth == 0 or game_over(game_state, map_data, mills):
        return evaluate(game_state, point_from, point_to, depth, difficulty, map_data, mills)
    
    possible_moves = get_possible_moves(game_state, map_data, mills)
    
    # Move ordering
    possible_moves.sort(key=lambda move : evaluate(move[0], move[1], move[2], depth, difficulty, map_data, mills), reverse=maximizing_player)

    if maximizing_player:
        max_eval = float('-inf')

        for child_state, child_move_from, child_move_to in possible_moves:
            eval = minimax(child_state, child_move_from, child_move_to, depth - 1, difficulty, alpha, beta, False, map_data, mills, start_time, timeout)
            max_eval = max(max_eval, eval)

            if time.time() - start_time > timeout:
                break

            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        # Store the result and the depth in the memoization dictionary
        memo[memo_key] = (max_eval, depth, difficulty)

        return max_eval
    else:
        min_eval = float('inf')

        for child_state, child_move_from, child_move_to in possible_moves:
            eval = minimax(child_state, child_move_from, child_move_to, depth - 1, difficulty, alpha, beta, True, map_data, mills, start_time, timeout)
            min_eval = min(min_eval, eval)

            if time.time() - start_time > timeout:
                break

            beta = min(beta, eval)
            if beta <= alpha:
                break

        # Store the result and the depth in the memoization dictionary
        memo[memo_key] = (min_eval, depth, difficulty)

        return min_eval

def game_over(game_state: TGameState, map_data: TMapData, mills: TMills) -> bool:
    if count_pieces(game_state, 'black') < 3 or count_pieces(game_state, 'white') < 3:
        return True
    
    # Check if the player has any possible moves
    return len(get_possible_moves(game_state, map_data, mills)) == 0

def count_pieces(game_state: TGameState, player: TPlayer) -> int:
    pieces_on_board = len([point for point in game_state['occupiedPoints'] if point['player'] == player])
    unplaced_pieces = game_state['unplacedPieces'][game_state['player']]
    return pieces_on_board + unplaced_pieces

def evaluate(game_state: TGameState, point_from: TPoint, point_to: TPoint, depth: int, difficulty: TDifficulty, map_data: TMapData, mills: TMills) -> float:
    strategic_value = float(0)
    opponent = get_opponent(game_state)
    occupied_points = {point['point']: point['player'] for point in game_state['occupiedPoints']}

    # Consider the points point_from and point_to
    if is_part_of_mill(point_to, game_state, mills):
        strategic_value += 331 * (1 + depth / 10) # Very high value for forming a mill

    # Consider the number of pieces each player has
    strategic_value += count_pieces(game_state, game_state['player']) * 20 * (1 + depth / 10)
    strategic_value -= count_pieces(game_state, opponent) * 20 * (1 + depth / 10)
        
    if (difficulty == 'hard'):
        strategic_value += calculate_piece_coordination(game_state, map_data) * 10 * (1 + depth / 10)

        # Penalize risky moves that expose a piece
        strategic_value -= penalize_risky_moves(game_state, map_data, mills) * 12 * (1 + depth / 10)

        # Check if a move broke player's own mill
        if point_from != None and is_part_of_mill(point_from, game_state, mills):
            if can_form_mill(point_to, game_state, map_data, mills):
                strategic_value += 20 * (1 + depth / 10) # Add value for forming a mill after breaking own mill
            else:
                strategic_value -= 10 * (1 + depth / 10) # Subtract value for breaking own mill without chance of forming a mill

    if (difficulty == 'medium' or difficulty == 'hard'):
        # Check if a move broke opponent's mill
        if is_part_of_mill({'point': point_to['point'], 'player': opponent}, game_state, mills):
            strategic_value += 30 * (1 + depth / 10)

        strategic_value += calculate_winning_configuration(game_state, map_data) * 1000 * (1 + depth / 10)

    return strategic_value

def calculate_winning_configuration(game_state: TGameState, map_data: TMapData) -> int:
    player = game_state['player']
    opponent = get_opponent(game_state)

    if count_pieces(game_state, opponent) == 2:
        return 1  # The opponent has been reduced to two pieces, so it's a winning configuration for the player
    elif not has_legal_move(game_state, opponent, map_data):
        return 1  # The opponent has no legal moves, so it's a winning configuration for the player
    elif count_pieces(game_state, player) == 2:
        return -1  # The player has been reduced to two pieces, so it's a losing configuration
    elif not has_legal_move(game_state, player, map_data):
        return -1  # The player has no legal moves, so it's a losing configuration
    else:
        return 0  # Neither side has won or lost yet
    
def has_legal_move(game_state: TGameState, player: TPlayer, map_data: TMapData) -> bool:
    if can_player_place(game_state):
        # Check if there are any empty points for placing a piece
        return any(not is_point_taken(game_state, {'point': point, 'player': player}) for point in map_data['points'])
    elif can_player_jump(game_state):
        # Check if there are any legal jumps for the player
        return any(
            is_valid_move(game_state, {'point': point_to, 'from': point_from}, map_data)
            for point_from in game_state['occupiedPoints']
            if point_from['player'] == player
            for point_to in map_data['points']
            if not is_point_taken(game_state, {'point': point_to, 'player': player})
        )
    else:
        # Check if there are any legal movements for the player
        return any(
            is_valid_move(game_state, {'point': point_to, 'from': point_from}, map_data)
            for point_from in game_state['occupiedPoints']
            if point_from['player'] == player
            for point_to in map_data['points']
            if not is_point_taken(game_state, {'point': point_to, 'player': player})
        )

# In the context of this function, "coordination" refers to the number of direct connections a piece has to other pieces owned by the same player.
# The function iterates over all points occupied by pieces on the board.
# For each point occupied by the current player, it finds all directly connected points (i.e., points that can be reached in one move).
# It then counts how many of these points are also occupied by the current player's pieces.
def calculate_piece_coordination(game_state: TGameState, map_data: TMapData) -> float:
    coordination_value = 0

    for occupied_point in game_state['occupiedPoints']:
        if occupied_point['player'] == game_state['player']:
            connections = next((point[1:] for point in map_data['connections'] if point[0] == occupied_point['point']), [])
            friendly_adjacent_pieces = sum(1 for point in connections if occupied_point.get(point, None) == game_state['player'])
            coordination_value += friendly_adjacent_pieces

    return coordination_value

# The function iterates over all points occupied by pieces on the board.
# For each point occupied by the current player, it finds all directly connected points (i.e., points that can be reached in one move).
# It then checks if any of these points are not occupied and are part of a mill.
# If such a point is found, it is considered a risky move and the penalty score is incremented.
def penalize_risky_moves(game_state: TGameState, map_data: TMapData, mills: TMills) -> float:
    risky_move_penalty = 0

    for occupied_point in game_state['occupiedPoints']:
        if occupied_point['player'] == game_state['player']:
            connections = next((point[1:] for point in map_data['connections'] if point[0] == occupied_point['point']), [])
            for point in connections:
                if not any(occupied_point['point'] == point for occupied_point in game_state['occupiedPoints']):
                    if is_part_of_mill({'point': point, 'player': game_state['player']}, game_state, mills):
                        # Penalize moves that expose a piece forming part of a mill
                        risky_move_penalty += 1

    return risky_move_penalty

def can_form_mill(point: TPoint, game_state: TGameState, map_data: TMapData, mills: TMills) -> bool:
    player = game_state['player']
    occupied_points = {point['point']: point['player'] for point in game_state['occupiedPoints']}

    # Check if there's a mill that includes the given point
    for mill in mills[point['point']]:
        # Check if all points in the mill are occupied by the player or are the given point
        if all(occupied_points.get(p, None) == player or p == point for p in mill):
            # Check if there's another piece of the same player that can move to the given point
            for p in mill:
                if p != point:
                    if can_player_jump(game_state):
                        # In the jumping phase, a piece can move to any vacant point
                        if occupied_points.get(p, None) == player:
                            return True
                    else:
                        # Get the connections for the point
                        connections = next((point[1:] for point in map_data['connections'] if point[0] == p), [])
                        # Check if the given point is a connection
                        if point in connections and occupied_points.get(p, None) == player:
                            return True

    return False

def is_valid_move(game_state: TGameState, move: TPoint, map_data: TMapData) -> bool:
    if is_point_taken(game_state, move):
        return False

    if can_player_place(game_state):
        return True

    if is_moving_piece(game_state, move, map_data):
        return True

    if can_player_jump(game_state):
        return True

    return False

def is_point_taken(game_state: TGameState, point: TPoint) -> bool:
    return any(occupied_point['point'] == point['point'] for occupied_point in game_state['occupiedPoints'])

def is_moving_piece(game_state: TGameState, move: TPoint, map_data: TMapData):
    if 'from' not in move:
        return False

    if not any(point['point'] == move['from'] and point['player'] == game_state['player'] for point in game_state['occupiedPoints']):
        return False

    if not any(connection[0] == move['from'] and move['point'] in connection[1:] for connection in map_data['connections']):
        return False
    
    return game_state['unplacedPieces'][game_state['player']] == 0

def get_opponent(game_state: TGameState) -> TPlayer:
    return 'white' if game_state['player'] == 'black' else 'black'

def switch_player(game_state: TGameState):
    game_state['player'] = get_opponent(game_state)

def can_player_place(game_state: TGameState) -> bool:
    return game_state['unplacedPieces'][game_state['player']] > 0

def can_player_jump(game_state: TGameState) -> bool:
    return game_state['player'] == 'black' and count_pieces(game_state, 'black') == 3 or game_state['player'] == 'white' and count_pieces(game_state, 'white') == 3

def get_possible_moves(game_state: TGameState, map_data: TMapData, mills: TMills) -> List[TGameState]:
    if can_player_place(game_state):
        return get_possible_placement_moves(game_state, map_data, mills)
    elif can_player_jump(game_state):
        return get_possible_jumping_moves(game_state, map_data, mills)
    else:
        return get_possible_movement_moves(game_state, map_data, mills)

def get_possible_placement_moves(game_state: TGameState, map_data: TMapData, mills: TMills) -> List[TGameState]:
    possible_moves = []
    points_taken = set(point['point'] for point in game_state['occupiedPoints'])

    for point in map_data['points']:
        if point not in points_taken:
            new_game_state = generate_new_game_state_for_placement(game_state, point)

            if is_valid_move(game_state, {'point': point}, map_data):
                new_point: TPoint = {'point': point, 'player': game_state['player']}

                if is_part_of_mill(new_point, new_game_state, mills):
                    remove_opponents_piece(new_game_state, mills)

                switch_player(new_game_state)
                possible_moves.append((new_game_state, None, new_point))

    return possible_moves

def get_possible_movement_moves(game_state: TGameState, map_data: TMapData, mills: TMills) -> List[TGameState]:
    # Generate all possible moves for moving a point to an adjacent spot
    possible_moves = []

    for occupied_point in [occupied_point for occupied_point in game_state['occupiedPoints'] if occupied_point['player'] == game_state['player']]:
        connections = next((point[1:] for point in map_data['connections'] if point[0] == occupied_point['point']), [])
        for point in connections:
            if point and not any(occupied_point['point'] == point for occupied_point in game_state['occupiedPoints']):
                new_game_state = generate_new_game_state_for_movement(game_state, occupied_point, point)

                if is_valid_move(game_state, {'point': point, 'from': occupied_point['point']}, map_data):
                    if is_part_of_mill({'point': point, 'player': game_state['player']}, new_game_state, mills):
                        remove_opponents_piece(new_game_state, mills)

                    new_point = {'point': point, 'player': game_state['player']}
                    # old_point = {'point': occupied_point['point'], 'player': game_state['player']}
                    switch_player(new_game_state)
                    possible_moves.append((new_game_state, occupied_point, new_point))

    return possible_moves

def get_possible_jumping_moves(game_state: TGameState, map_data: TMapData, mills: TMills) -> List[TGameState]:
    # Generate all possible moves for jumping a point to any empty spot
    possible_moves = []
    points_taken = set(point['point'] for point in game_state['occupiedPoints'])

    for occupied_point in [occupied_point for occupied_point in game_state['occupiedPoints'] if occupied_point['player'] == game_state['player']]:
        for point in map_data['points']:
            if point not in points_taken:
                new_game_state = generate_new_game_state_for_movement(game_state, occupied_point, point)

                if is_valid_move(game_state, {'point': point, 'from': occupied_point['point']}, map_data):
                    if is_part_of_mill({'point': point, 'player': game_state['player']}, new_game_state, mills):
                        remove_opponents_piece(new_game_state, mills)

                    new_point = {'point': point, 'player': game_state['player']}
                    # old_point = {'point': occupied_point['point'], 'player': game_state['player']}
                    switch_player(new_game_state)
                    possible_moves.append((new_game_state, occupied_point, new_point))

    return possible_moves

def generate_new_game_state_for_placement(game_state: TGameState, point: TPoint) -> TGameState :
    # Generate a new game state for placing a point on an empty spot
    new_game_state = game_state.copy()
    new_points_taken = new_game_state['occupiedPoints'].copy()
    new_points_taken.append({'point': point, 'player': game_state['player']})
    new_game_state['occupiedPoints'] = new_points_taken

    new_unplaced_pieces = new_game_state['unplacedPieces'].copy()
    new_unplaced_pieces[game_state['player']] -= 1
    new_game_state['unplacedPieces'] = new_unplaced_pieces

    return new_game_state

def generate_new_game_state_for_movement(game_state: TGameState, occupied_point: TPoint, point: TPoint) -> TGameState:
    # Generate a new game state for moving a point to an adjacent spot
    new_game_state = game_state.copy()
    new_points_taken = [p for p in new_game_state['occupiedPoints'] if p != occupied_point]
    new_points_taken.append({'point': point, 'player': game_state['player']})
    new_game_state['occupiedPoints'] = new_points_taken

    return new_game_state

def remove_opponents_piece(game_state: TGameState, mills: TMills):
    # Get the opponent's pieces that are not part of a mill
    non_mill_pieces = [occupied_point for occupied_point in game_state['occupiedPoints'] if occupied_point['player'] == get_opponent(game_state) and not is_part_of_mill(occupied_point, game_state, mills)]

    # If all of the opponent's pieces are part of a mill, then any piece can be removed
    if not non_mill_pieces:
        non_mill_pieces = [occupied_point for occupied_point in game_state['occupiedPoints'] if occupied_point['player'] == get_opponent(game_state)]

    # Remove one of the opponent's pieces
    if non_mill_pieces:
        game_state['occupiedPoints'].remove(non_mill_pieces[0])

def is_part_of_mill(piece, game_state: TGameState, mills: TMills) -> bool:
    if piece['point'] not in [point['point'] for point in game_state['occupiedPoints']]:
        return False
        
    for mill in mills[piece['point']]:
        if all(any(occ_point['point'] == point and occ_point['player'] == piece['player'] for occ_point in game_state['occupiedPoints']) for point in mill):
            return True
        
    return False

def precalculate_mills(map_data: TMapData) -> TMills:
    mills = {point: [] for point in map_data['points']}
    for mill in map_data['mills']:
        if len(mill) == 3:  # A mill is formed by 3 points
            for point in mill:
                mills[point].append(set(mill))
    return mills

def load_map_file(map_name: str) -> TMapData:
    with open(os.path.join(os.path.dirname(__file__), 'maps', map_name + '.json')) as f:
        return json.load(f)
