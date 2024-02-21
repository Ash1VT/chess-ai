import io
import pickle
import sys

import cairosvg
import chess
import chess.variant
import pygame
from keras.saving.save import load_model
from pygame import Color, Rect, Surface
from pygame.font import Font

from config import FIGURES, SQUARE_SIZE, BOARD_SQUARES_COUNT, LIGHT_SQUARE_COLOR, DARK_SQUARE_COLOR, BOARD_HEIGHT, \
    BOARD_WIDTH, STATISTICS_WIDTH, STATISTICS_HEIGHT, DISPLAY_HEIGHT, DISPLAY_WIDTH, SELECTED_SQUARE_COLOR, \
    MODEL_FILE_PATH, TOKENIZER_FILE_PATH
from prediction import predict_outcome

# Загрузка шахматных изображений
IMAGES = {}

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
numbers = ['8', '7', '6', '5', '4', '3', '2', '1']


def get_full_figure_image_path(figure_filename: str):
    return f'../images/{figure_filename}'


for figure_code, figure_filename in FIGURES.items():
    png_bytes = cairosvg.svg2png(url=get_full_figure_image_path(figure_filename), output_width=SQUARE_SIZE,
                                 output_height=SQUARE_SIZE)
    IMAGES[figure_code] = pygame.image.load(io.BytesIO(png_bytes))


def load_tokenizer():
    with open(TOKENIZER_FILE_PATH, 'rb') as file:
        return pickle.load(file)


def display_text_from_new_line_centered(screen, font, text, rect, color):
    lines = text.split('\n')
    y_offset = 0
    for line in lines:
        rendered_text = font.render(line, True, color)
        text_rect = rendered_text.get_rect(center=rect.center)
        text_rect.y += y_offset
        screen.blit(rendered_text, text_rect)
        y_offset += rendered_text.get_height()


# Функция отрисовки шахматной доски
def draw_board(screen: Surface, board, font):
    for y in range(BOARD_SQUARES_COUNT):
        for x in range(BOARD_SQUARES_COUNT):
            color = LIGHT_SQUARE_COLOR if (x + y) % 2 == 0 else DARK_SQUARE_COLOR
            pygame.draw.rect(screen, color,
                             pygame.Rect(x * SQUARE_SIZE + SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            piece = board.piece_at(chess.square(x, 7 - y))
            if piece:
                screen.blit(IMAGES[piece.symbol()],
                            pygame.Rect(x * SQUARE_SIZE + SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    # Отрисовка буквенно-цифровых кодов клеток
    for i in range(8):
        text_surface = font.render(letters[i], True, (255, 255, 255))
        text_rect = text_surface.get_rect(
            center=Rect(i * SQUARE_SIZE + SQUARE_SIZE, BOARD_HEIGHT - SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE).center)
        screen.blit(text_surface, text_rect)
        text_surface = font.render(numbers[i], True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=Rect(0, i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE).center)
        screen.blit(text_surface, text_rect)


def draw_statictics(screen, board, moves: str, win_percentage: str, font: Font):
    screen.fill((Color(255, 255, 255, 0)), Rect(BOARD_WIDTH, 0, STATISTICS_WIDTH, STATISTICS_HEIGHT))
    pygame.draw.rect(screen, Color(0, 0, 0), Rect(BOARD_WIDTH, 0, STATISTICS_WIDTH, STATISTICS_HEIGHT), 1)

    statistics_text = ""
    # Отображение текста о текущем ходе
    if board.turn == chess.WHITE:
        statistics_text += 'Turn: White\n'
    else:
        statistics_text += 'Turn: Black\n'

    if moves:
        statistics_text += f'Chances for win: {win_percentage}%\n'

    display_text_from_new_line_centered(screen, font, statistics_text, Rect(BOARD_WIDTH, 0, STATISTICS_WIDTH, STATISTICS_HEIGHT), (0, 0, 0))
    # text_surface = font.render(statistics_text, True, (0, 0, 0))
    # text_rect = text_surface.get_rect(center=Rect(BOARD_WIDTH, 0, STATISTICS_WIDTH, STATISTICS_HEIGHT).center)
    # screen.blit(text_surface, text_rect)


def to_algebraic_notation(board: chess.Board, move: chess.Move) -> str:
    color = 'W' if board.turn else 'B'
    turn_number = board.fullmove_number
    san = board.san(move)

    return f"{color}{turn_number}.{san}"


# Основная функция программы
def main():
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption('Chess')
    clock = pygame.time.Clock()
    board = chess.Board()
    font = pygame.font.Font("../fonts/Roboto-Regular.ttf", 20)
    moves_history = ""
    model = load_model(MODEL_FILE_PATH)
    win_percentage = ""
    tokenizer = load_tokenizer()

    selected_square = {
        'number': None,
        'name': None
    }

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                square_number = chess.square((x - SQUARE_SIZE) // SQUARE_SIZE, 7 - y // SQUARE_SIZE)
                square_name = chess.square_name(square_number)
                if selected_square["number"] is not None:
                    move = chess.Move(selected_square["number"], square_number)
                    promotion_move = chess.Move(selected_square["number"], square_number, promotion=chess.QUEEN)
                    if move in board.legal_moves:
                        if moves_history:
                            moves_history += " "
                        moves_history += board.san(move)
                        board.push(move)
                        print(moves_history)
                        win_percentage = predict_outcome(moves_history, tokenizer, model)
                    elif promotion_move in board.legal_moves:
                        if moves_history:
                            moves_history += " "
                        moves_history += board.san(promotion_move)
                        board.push(promotion_move)
                        print(moves_history)
                        win_percentage = predict_outcome(moves_history, tokenizer, model)

                if selected_square["name"] == square_name:
                    selected_square["number"] = None
                    selected_square["name"] = None
                elif square_name in chess.SQUARE_NAMES:
                    selected_square["number"] = square_number
                    selected_square["name"] = square_name

        screen.fill((0, 0, 0))
        draw_board(screen, board, font)
        draw_statictics(screen, board, moves_history, win_percentage, font)

        if selected_square["number"] is not None:
            x, y = chess.square_file(selected_square["number"]), 7 - chess.square_rank(selected_square["number"])
            pygame.draw.rect(screen, SELECTED_SQUARE_COLOR,
                             pygame.Rect(x * SQUARE_SIZE + SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

        pygame.display.flip()
        clock.tick(30)


if __name__ == '__main__':
    main()
