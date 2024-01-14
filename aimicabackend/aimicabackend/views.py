from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import os
from aimicabackend.mica import get_best_move

@require_http_methods(["GET", "POST"])
@csrf_exempt
def OK(request):
    return HttpResponse("OK")

@require_http_methods(["GET", "POST"])
@csrf_exempt
def maps_list(request):
    current_dir = os.path.dirname(__file__)
    maps_dir = os.path.join(current_dir, 'maps')

    maps = []

    for file in os.listdir(maps_dir):
        if file.endswith('.json'):
            maps.append({
                'map_name': file[:-5],
                'map_data': json.load(open(os.path.join(maps_dir, file)))
            })

    return HttpResponse(json.dumps(maps))


@require_http_methods(["GET", "POST"])
@csrf_exempt
def get_move(request):
    body = request.body

    map_name = json.loads(body)['mapName']
    depth = json.loads(body)['depth']
    difficulty = json.loads(body)['difficulty']
    current_game_state = json.loads(body)['gameState']

    new_game_state = get_best_move(current_game_state, depth, map_name)
    
    return HttpResponse(json.dumps(new_game_state))

# Game map example:
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
