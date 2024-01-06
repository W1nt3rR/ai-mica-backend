from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
from aimicabackend.mica import calculateMove

@require_http_methods(["GET", "POST"])
@csrf_exempt
def OK(request):
    return HttpResponse("OK")

@require_http_methods(["GET", "POST"])
@csrf_exempt
def maps_list(request):
    return HttpResponse("map_list")

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
