from typing import Dict, List, TypedDict, Optional, Literal, Set

TPlayer = Literal["black", "white"]
TMills = Dict[str, List[Set[str]]]

class TPoint(TypedDict):
    point: str
    player: TPlayer

class TGameState(TypedDict):
    player: TPlayer
    unplacedPieces: Dict[str, int]
    occupiedPoints: List[TPoint]

class TMapData(TypedDict):
    points: List[str]
    mills: List[List[str]]
    connections: List[List[str]]
