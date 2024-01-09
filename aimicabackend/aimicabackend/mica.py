import json
import os

def minimax(game_state, depth, alpha, beta, maximizing_player, map_data):
    if depth == 0 or game_over(game_state, map_data):
        return evaluate(game_state, map_data)

    if maximizing_player:
        max_eval = float('-inf')

        for child in get_possible_moves(game_state, map_data):
            eval = minimax(child, depth - 1, alpha, beta, False, map_data)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)

            if beta <= alpha:
                break

        return max_eval
    else:
        min_eval = float('inf')

        for child in get_possible_moves(game_state, map_data):
            eval = minimax(child, depth - 1, alpha, beta, True, map_data)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)

            if beta <= alpha:
                break

        return min_eval

def get_best_move(game_state, depth, map_data):
    max_eval = float('-inf')
    best_move = None

    for move in get_possible_moves(game_state, map_data):
        eval = minimax(move, depth, float('-inf'), float('inf'), False, map_data)

        if eval > max_eval:
            max_eval = eval
            best_move = move

    return best_move

def game_over(game_state, map_data):
    no_points_left = game_state['unplacedPiecesBlack'] == 0 or game_state['unplacedPiecesWhite'] == 0
    no_legal_moves = len(get_possible_moves(game_state, map_data)) == 0
    return no_points_left or no_legal_moves

def evaluate(game_state, map_data):
    points_difference = game_state['unplacedPiecesBlack'] - game_state['unplacedPiecesWhite']
    mills_difference = count_mills(game_state, 'black', map_data) - count_mills(game_state, 'white', map_data)
    
    # Calculate the number of possible moves for each player
    possible_moves_black = len(get_possible_moves({'player': 'black', 'unplacedPiecesBlack': game_state['unplacedPiecesBlack'], 'unplacedPiecesWhite': game_state['unplacedPiecesWhite'], 'occupiedPoints': game_state['occupiedPoints']}, map_data))
    possible_moves_white = len(get_possible_moves({'player': 'white', 'unplacedPiecesBlack': game_state['unplacedPiecesBlack'], 'unplacedPiecesWhite': game_state['unplacedPiecesWhite'], 'occupiedPoints': game_state['occupiedPoints']}, map_data))
    possible_moves_difference = possible_moves_black - possible_moves_white

    # Calculate the number of pieces on the board for each player
    pieces_black = len([point for point in game_state['occupiedPoints'] if point['player'] == 'black'])
    pieces_white = len([point for point in game_state['occupiedPoints'] if point['player'] == 'white'])
    pieces_difference = pieces_black - pieces_white

    # Calculate the strategic value of the move
    strategic_value = 0
    if is_placing_piece(game_state):
        strategic_value = calculate_strategic_value(game_state, map_data)

    # You might want to adjust the weights depending on what you consider to be more important
    return points_difference + 2 * mills_difference + 0.5 * possible_moves_difference + pieces_difference # + strategic_value

def calculate_strategic_value(game_state, map_data):
    strategic_value = 0
    opponent = 'white' if game_state['player'] == 'black' else 'black'

    for point in map_data['points']:
        potential_game_state = game_state.copy()
        potential_game_state['occupiedPoints'].append({'point': point, 'player': game_state['player']})

        if is_part_of_mill({'point': point, 'player': game_state['player']}, potential_game_state, map_data):
            strategic_value += 100  # High value for forming a mill
        elif is_part_of_mill({'point': point, 'player': opponent}, potential_game_state, map_data):
            strategic_value += 50  # Medium value for blocking a mill
        else:
            strategic_value += 1  # Low value for all other points

    return strategic_value

def count_mills(game_state, color, map_data):
    count = 0
    for mill in map_data['mills']:
        if all(point in [occupied_point['point'] for occupied_point in game_state['occupiedPoints'] if occupied_point['player'] == color] for point in mill):
            count += 1
    return count

def is_valid_move(game_state, move, map_data):
    if is_point_taken(game_state, move):
        return False

    if is_placing_piece(game_state):
        return True

    if is_moving_piece(game_state, move, map_data):
        return True

    if is_moving_anywhere(game_state):
        return True

    return False

def is_point_taken(game_state, move):
    return any(occupied_point['point'] == move['point'] for occupied_point in game_state['occupiedPoints'])

def is_placing_piece(game_state):
    return game_state['player'] == 'black' and game_state['unplacedPiecesBlack'] > 0 or game_state['player'] == 'white' and game_state['unplacedPiecesWhite'] > 0

def is_moving_piece(game_state, move, map_data):
    if 'from' not in move:
        return False

    if not any(point['point'] == move['from'] and point['player'] == game_state['player'] for point in game_state['occupiedPoints']):
        return False

    if not any(connection[0] == move['from'] and move['point'] in connection[1:] for connection in map_data['connections']):
        return False

    return (game_state['player'] == 'black' and game_state['unplacedPiecesBlack'] == 0) or (game_state['player'] == 'white' and game_state['unplacedPiecesWhite'] == 0)

def is_moving_anywhere(game_state):
    pieces_black = len([point for point in game_state['occupiedPoints'] if point['player'] == 'black'])
    pieces_white = len([point for point in game_state['occupiedPoints'] if point['player'] == 'white'])

    return (game_state['player'] == 'black' and pieces_black == 3) or (game_state['player'] == 'white' and pieces_white == 3)

def switch_player(game_state):
    game_state['player'] = 'white' if game_state['player'] == 'black' else 'black'

def get_possible_moves(game_state, map_data):
    possible_moves = []

    if game_state['player'] == 'black' and game_state['unplacedPiecesBlack'] > 0 or game_state['player'] == 'white' and game_state['unplacedPiecesWhite'] > 0:
        possible_moves.extend(get_possible_placement_moves(game_state, map_data))
    else:
        possible_moves.extend(get_possible_movement_moves(game_state, map_data))

    return possible_moves

def get_possible_placement_moves(game_state, map_data):
    # Generate all possible moves for placing a point on an empty spot
    possible_moves = []
    points_taken = set(point['point'] for point in game_state['occupiedPoints'])

    for point in map_data['points']:
        if point not in points_taken:
            new_game_state = generate_new_game_state_for_placement(game_state, point)
            if is_valid_move(game_state, {'point': point}, map_data):
                if is_part_of_mill({'point': point, 'player': game_state['player']}, new_game_state, map_data):
                    remove_opponents_piece(new_game_state, map_data)
                switch_player(new_game_state)
                possible_moves.append(new_game_state)

    return possible_moves

def get_possible_movement_moves(game_state, map_data):
    # Generate all possible moves for moving a point to an adjacent spot
    possible_moves = []

    for occupied_point in [occupied_point for occupied_point in game_state['occupiedPoints'] if occupied_point['player'] == game_state['player']]:
        connections = next((point[1:] for point in map_data['connections'] if point[0] == occupied_point['point']), [])
        for point in connections:
            if point and not any(occupied_point['point'] == point for occupied_point in game_state['occupiedPoints']):
                new_game_state = generate_new_game_state_for_movement(game_state, occupied_point, point)
                if is_valid_move(game_state, {'point': point, 'from': occupied_point['point']}, map_data):
                    if is_part_of_mill({'point': point, 'player': game_state['player']}, new_game_state, map_data):
                        remove_opponents_piece(new_game_state, map_data)
                    switch_player(new_game_state)
                    possible_moves.append(new_game_state)

    return possible_moves

def generate_new_game_state_for_placement(game_state, point):
    # Generate a new game state for placing a point on an empty spot
    new_game_state = game_state.copy()
    new_points_taken = new_game_state['occupiedPoints'].copy()
    new_points_taken.append({'point': point, 'player': game_state['player']})
    new_game_state['occupiedPoints'] = new_points_taken

    if game_state['player'] == 'black':
        new_game_state['unplacedPiecesBlack'] -= 1
    else:
        new_game_state['unplacedPiecesWhite'] -= 1

    return new_game_state

def generate_new_game_state_for_movement(game_state, occupied_point, point):
    # Generate a new game state for moving a point to an adjacent spot
    new_game_state = game_state.copy()
    new_points_taken = [p for p in new_game_state['occupiedPoints'] if p != occupied_point]
    new_points_taken.append({'point': point, 'player': game_state['player']})
    new_game_state['occupiedPoints'] = new_points_taken

    return new_game_state

def remove_opponents_piece(game_state, map_data):
    # Get the opponent's color
    opponent_color = 'white' if game_state['player'] == 'black' else 'black'

    # Get the opponent's pieces that are not part of a mill
    non_mill_pieces = [occupied_point for occupied_point in game_state['occupiedPoints'] if occupied_point['player'] == opponent_color and not is_part_of_mill(occupied_point, game_state, map_data)]

    # If all of the opponent's pieces are part of a mill, then any piece can be removed
    if not non_mill_pieces:
        non_mill_pieces = [occupied_point for occupied_point in game_state['occupiedPoints'] if occupied_point['player'] == opponent_color]

    # Remove one of the opponent's pieces
    if non_mill_pieces:
        game_state['occupiedPoints'].remove(non_mill_pieces[0])

def is_part_of_mill(occupied_point, game_state, map_data):
    # Check if a piece is part of a mill
    for mill in map_data['mills']:
        if occupied_point['point'] in mill and all(point in [occupied_point['point'] for occupied_point in game_state['occupiedPoints'] if occupied_point['player'] == game_state['player']] for point in mill):
            return True
    return False

def calculateMove(mapName, difficulty, game_state):
    # Get the best move

    # Load game state and map from JSON
    with open(os.path.join(os.path.dirname(__file__), 'maps', mapName + '.json')) as f:
        map_data = json.load(f)

    #
    best_move = get_best_move(game_state, getDepthByDifficulty(difficulty), map_data)

    return best_move

def getDepthByDifficulty(difficulty):
    if difficulty == "easy":
        return 2
    elif difficulty == "medium":
        return 5
    elif difficulty == "hard":
        return 7
    else:
        return 1


#  0-----------1-----------2
#  |           |           |
#  |   3-------4-------5   |
#  |   |       |       |   |
#  |   |   6---7---8   |   |
#  |   |   |       |   |   |
#  9--10--11       12--13--14
#  |   |   |       |   |   |
#  |   |   15-16--17   |   |
#  |   |       |       |   |
#  |   18------19-----20   |
#  |           |           |
#  21---------22----------23
