from typing import Optional, List, Tuple


def parse_move(move: str) -> Tuple[str, int, str]:
    splitted_move = move.split('.')
    move_title = splitted_move[0]
    move_description = splitted_move[1]

    move_color = move_title[0]
    move_turn = int(move_title[1:])
    return move_color, move_turn, move_description


def process_move(move: str, turn_count: float) -> Optional[str]:
    if not move:
        return
    move_color, move_turn, move_description = parse_move(move)
    if not move_color or not move_turn or not move_description:
        return

    if move_turn != int(turn_count):
        return

    return move_description


def process_line(line: str) -> Optional[List]:
    turn_count = 1

    row_data = line.split()

    moves = row_data[17:]

    if not moves:
        return

    processed_moves = []
    for move in moves:
        processed_move = process_move(move, turn_count)

        if not processed_move:
            return

        processed_moves.append(processed_move)
        turn_count += 0.5

    moves = ' '.join(processed_moves)
    if not moves:
        return

    result = row_data[2]

    if result == "1-0":
        result = 1
    elif result == "1/2-1/2" or result == "0-1":
        result = 0
    else:
        return

    return [result, moves]
