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
    # Game is over if either player has no points left or cannot make a legal move
    no_points_left = game_state['pointsLeftoverBlack'] == 0 or game_state['pointsLeftoverWhite'] == 0
    no_legal_moves = len(get_possible_moves(game_state, map_data)) == 0
    return no_points_left or no_legal_moves

def evaluate(game_state, map_data):
    # More sophisticated evaluation function: difference in number of points, plus number of mills
    points_difference = game_state['pointsLeftoverBlack'] - game_state['pointsLeftoverWhite']
    mills_difference = count_mills(game_state, 'black', map_data) - count_mills(game_state, 'white', map_data)
    return points_difference + mills_difference

def count_mills(game_state, color, map_data):
    # Count the number of mills for a given color
    count = 0
    for mill in map_data['mills']:
        if all(point in [taken_point['point'] for taken_point in game_state['pointsTaken'] if taken_point['color'] == color] for point in mill):
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
    return any(taken_point['point'] == move['point'] for taken_point in game_state['pointsTaken'])

def is_placing_piece(game_state):
    return game_state['playerTurn'] == 'black' and game_state['pointsLeftoverBlack'] > 0 or game_state['playerTurn'] == 'white' and game_state['pointsLeftoverWhite'] > 0

def is_moving_piece(game_state, move, map_data):
    return 'from' in move and game_state['playerTurn'] == 'black' and game_state['pointsLeftoverBlack'] == 0 or game_state['playerTurn'] == 'white' and game_state['pointsLeftoverWhite'] == 0 and move['from'] not in map_data['adjacencies'][move['point']]

def is_moving_anywhere(game_state):
    return game_state['playerTurn'] == 'black' and len([taken_point for taken_point in game_state['pointsTaken'] if taken_point['color'] == 'black']) == 3 or game_state['playerTurn'] == 'white' and len([taken_point for taken_point in game_state['pointsTaken'] if taken_point['color'] == 'white']) == 3

def get_possible_moves(game_state, map_data):
    # Generate all possible moves: placing a point on an empty spot, or moving a point to an adjacent spot
    possible_moves = []

    for point in map_data['points']:
        if not any(taken_point['point'] == point for taken_point in game_state['pointsTaken']):
            new_game_state = game_state.copy()
            new_points_taken = new_game_state['pointsTaken'].copy()
            new_points_taken.append({'point': point, 'color': game_state['playerTurn']})
            new_game_state['pointsTaken'] = new_points_taken

            if game_state['playerTurn'] == 'black':
                new_game_state['pointsLeftoverBlack'] -= 1
            else:
                new_game_state['pointsLeftoverWhite'] -= 1

            if (is_valid_move(game_state, {'point': point}, map_data)):
                # Check if a mill is formed
                if is_part_of_mill({'point': point, 'color': game_state['playerTurn']}, new_game_state, map_data):
                    remove_opponents_piece(new_game_state, map_data)
                possible_moves.append(new_game_state)

    if game_state['pointsLeftoverBlack'] == 0 and game_state['playerTurn'] == 'black' or game_state['pointsLeftoverWhite'] == 0 and game_state['playerTurn'] == 'white':
        for taken_point in [taken_point for taken_point in game_state['pointsTaken'] if taken_point['color'] == game_state['playerTurn']]:
            for point in map_data['points']:
                if not any(taken_point['point'] == point for taken_point in game_state['pointsTaken']):
                    new_game_state = game_state.copy()
                    new_game_state['pointsTaken'].remove(taken_point)
                    new_game_state['pointsTaken'].append({'point': point, 'color': game_state['playerTurn']})

                    if (is_valid_move(game_state, {'point': point}, map_data)):
                        # Check if a mill is formed
                        if is_part_of_mill({'point': point, 'color': game_state['playerTurn']}, new_game_state, map_data):
                            remove_opponents_piece(new_game_state, map_data)
                        possible_moves.append(new_game_state)

    return possible_moves

def remove_opponents_piece(game_state, map_data):
    # Get the opponent's color
    opponent_color = 'white' if game_state['playerTurn'] == 'black' else 'black'

    # Get the opponent's pieces that are not part of a mill
    non_mill_pieces = [taken_point for taken_point in game_state['pointsTaken'] if taken_point['color'] == opponent_color and not is_part_of_mill(taken_point, game_state, map_data)]

    # If all of the opponent's pieces are part of a mill, then any piece can be removed
    if not non_mill_pieces:
        non_mill_pieces = [taken_point for taken_point in game_state['pointsTaken'] if taken_point['color'] == opponent_color]

    # Remove one of the opponent's pieces
    if non_mill_pieces:
        game_state['pointsTaken'].remove(non_mill_pieces[0])

def is_part_of_mill(taken_point, game_state, map_data):
    # Check if a piece is part of a mill
    for mill in map_data['mills']:
        if taken_point['point'] in mill and all(point in [taken_point['point'] for taken_point in game_state['pointsTaken'] if taken_point['color'] == game_state['playerTurn']] for point in mill):
            return True
    return False

def calculateMove(mapName, difficulty, game_state):
    # Get the best move

    # Load game state and map from JSON
    with open(os.path.join(os.path.dirname(__file__), 'maps', mapName + '.json')) as f:
        map_data = json.load(f)

    #
    best_move = get_best_move(game_state, getDepthByDifficulty(difficulty), map_data)

    # set the player turn
    if best_move['playerTurn'] == 'black':
        best_move['playerTurn'] = 'white'
    else:
        best_move['playerTurn'] = 'black'

    return best_move

def getDepthByDifficulty(difficulty):
    if difficulty == "easy":
        return 1
    elif difficulty == "medium":
        return 2
    elif difficulty == "hard":
        return 3
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
