from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import os
from aimicabackend.mica import calculateMove

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
def getMove(request):
    body = request.body

    mapName = json.loads(body)['mapName']
    difficulty = json.loads(body)['difficulty']
    gameState = json.loads(body)['gameState']

    # gameState has the following format:
    # {
    #     "pointsTaken": [
    #           { point: "A1", color: "black"} ],
    #           { point: "A4", color: "white"} ],
    #           ...
    #     ],
    #     "pointsLeftoverBlack": 9,
    #     "pointsLeftoverWhite": 9,
    #     "playerTurn": "black",
    # }
    #

    newGameState = calculateMove(mapName, difficulty, gameState)
    
    return HttpResponse(json.dumps(newGameState))
