import json
import os
import time
from typing import List
from aimicabackend.types import TGameState, TMapData, TPoint, TMills, TPlayer, TDifficulty

memo = {}

def get_best_move(game_state: TGameState, depth: int, map_name: str, difficulty: TDifficulty, timeout: int) -> TGameState:
    start_time = time.time()

    max_eval = float('-inf')
    best_move = None
    map_data = load_map_file(map_name)
    mills = precalculate_mills(map_data)

    print("\nMILLS", mills)

    for child_state, child_move_from, child_move_to in get_possible_moves(game_state, map_data, mills):
        eval = minimax(child_state, child_move_from, child_move_to, depth, float('-inf'), float('inf'), False, map_data, mills)

        if time.time() - start_time > float(timeout):
            print("Timeout expired!")
            return best_move
        
        print("\nEVAL", eval)

        if eval > max_eval:
            max_eval = eval
            best_move = child_state
            print("\nPOTENTIAL BEST MOVE", eval)

    print ("\nBEST MOVE", max_eval, best_move)
    return best_move

def minimax(game_state: TGameState, point_from: TPoint, point_to: TPoint, depth: int, alpha: float, beta: float, maximizing_player: bool, map_data: TMapData, mills: TMills) -> float:
    # Convert the game state to a string
    game_state_str = json.dumps(game_state, sort_keys=True)
    point_from_str = json.dumps(point_from, sort_keys=True)
    point_to_str = json.dumps(point_to, sort_keys=True)
    memo_key = (game_state_str, point_from_str, point_to_str)

    # Check if the game state has already been evaluated
    if memo_key in memo:
        score, score_depth = memo[memo_key]
        if score_depth == depth:
            print("MEMOIZED score", score)
            return score

    if depth == 0 or game_over(game_state, map_data, mills):
        return evaluate(game_state, point_from, point_to, depth, map_data, mills)
    
    possible_moves = get_possible_moves(game_state, map_data, mills)
    
    # Move ordering
    possible_moves.sort(key=lambda move : evaluate(move[0], move[1], move[2], depth, map_data, mills), reverse=maximizing_player)

    if maximizing_player:
        max_eval = float('-inf')

        for child_state, child_move_from, child_move_to in possible_moves:
            eval = minimax(child_state, child_move_from, child_move_to, depth - 1, alpha, beta, False, map_data, mills)
            max_eval = max(max_eval, eval)

            alpha = max(alpha, eval)
            if beta <= alpha:
                # print("PRUNED", eval, alpha, beta)
                break

        # Store the result and the depth in the memoization dictionary
        memo[memo_key] = (max_eval, depth)

        # print("MAX EVAL", max_eval)
        return max_eval
    else:
        min_eval = float('inf')

        for child_state, child_move_from, child_move_to in possible_moves:
            eval = minimax(child_state, child_move_from, child_move_to, depth - 1, alpha, beta, True, map_data, mills)
            min_eval = min(min_eval, eval)

            beta = min(beta, eval)
            if beta <= alpha:
                # print("PRUNED", eval, alpha, beta)
                break

        # Store the result and the depth in the memoization dictionary
        memo[memo_key] = (min_eval, depth)

        # print("MIN EVAL", min_eval)
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

def evaluate(game_state: TGameState, point_from: TPoint, point_to: TPoint, depth: int, map_data: TMapData, mills: TMills) -> float:
    strategic_value = float(0)
    opponent = get_opponent(game_state)
    occupied_points = {point['point']: point['player'] for point in game_state['occupiedPoints']}

    # Consider the points point_from and point_to
    if is_part_of_mill(point_to, game_state, mills):
        strategic_value += 100 * (depth + 1) # (1 + (depth) / 10) # Very high value for forming a mill

    # Check if a move broke opponent's mill
    if is_part_of_mill({'point': point_to['point'], 'player': opponent}, game_state, mills):
        # print("BLOCKED OPPONENT'S MILL", point_to)
        strategic_value += 30

    # Check if a move broke player's own mill
    if point_from != None and is_part_of_mill(point_from, game_state, mills):
        if can_form_mill(point_to, game_state, map_data, mills):
            print("can form mill again")
            strategic_value += 20  # Add value for forming a mill after breaking own mill
        else:
            strategic_value -= 10  # Subtract value for breaking own mill without chance of forming a mill

    # Consider the number of pieces each player has
    strategic_value += count_pieces(game_state, game_state['player']) * 49
    strategic_value -= count_pieces(game_state, opponent) * 49

    return strategic_value # * (1 + (depth) / 10)

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
        # print("IS MOVING PIECE", move)
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
