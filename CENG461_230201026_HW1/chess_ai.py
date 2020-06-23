#I have included a report within the file I uploaded to the CMS, it contains all the explanations for the functions.
#Ali Görkem Yalçın - 230201026

import chess

pawnValue = 100
knightValue = 330
bishopValue = 340
rookValue = 530
queenValue = 970
kingValue = 10000000000


pawn_piece_square_table = [
0,  0,  0,  0,  0,  0,  0,  0,
50, 50, 50, 50, 50, 50, 50, 50,
10, 10, 20, 30, 30, 20, 10, 10,
 5,  5, 10, 25, 25, 10,  5,  5,
 0,  0,  0, 20, 20,  0,  0,  0,
 5, -5,-10,  0,  0,-10, -5,  5,
 5, 10, 10,-20,-20, 10, 10,  5,
 0,  0,  0,  0,  0,  0,  0,  0]

knight_piece_square_table = [
-50,-40,-30,-30,-30,-30,-40,-50,
-40,-20,  0,  0,  0,  0,-20,-40,
-30,  0, 10, 15, 15, 10,  0,-30,
-30,  5, 15, 20, 20, 15,  5,-30,
-30,  0, 15, 20, 20, 15,  0,-30,
-30,  5, 10, 15, 15, 10,  5,-30,
-40,-20,  0,  5,  5,  0,-20,-40,
-50,-40,-30,-30,-30,-30,-40,-50]

bishop_piece_square_table = [
-20,-10,-10,-10,-10,-10,-10,-20,
-10,  0,  0,  0,  0,  0,  0,-10,
-10,  0,  5, 10, 10,  5,  0,-10,
-10,  5,  5, 10, 10,  5,  5,-10,
-10,  0, 10, 10, 10, 10,  0,-10,
-10, 10, 10, 10, 10, 10, 10,-10,
-10,  5,  0,  0,  0,  0,  5,-10,
-20,-10,-10,-10,-10,-10,-10,-20]

rook_piece_square_table = [
 0,  0,  0,  0,  0,  0,  0,  0,
 5, 10, 10, 10, 10, 10, 10,  5,
-5,  0,  0,  0,  0,  0,  0, -5,
-5,  0,  0,  0,  0,  0,  0, -5,
-5,  0,  0,  0,  0,  0,  0, -5,
-5,  0,  0,  0,  0,  0,  0, -5,
-5,  0,  0,  0,  0,  0,  0, -5,
 0,  0,  0,  5,  5,  0,  0,  0]

queen_piece_square_table = [
-20,-10,-10, -5, -5,-10,-10,-20,
-10,  0,  0,  0,  0,  0,  0,-10,
-10,  0,  5,  5,  5,  5,  0,-10,
 -5,  0,  5,  5,  5,  5,  0, -5,
  0,  0,  5,  5,  5,  5,  0, -5,
-10,  5,  5,  5,  5,  5,  0,-10,
-10,  0,  5,  0,  0,  0,  0,-10,
-20,-10,-10, -5, -5,-10,-10,-20]

king_early_game_piece_square_table = [
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-20,-30,-30,-40,-40,-30,-30,-20,
-10,-20,-20,-20,-20,-20,-20,-10,
 20, 20,  0,  0,  0,  0, 20, 20,
 20, 30, 10,  0,  0, 10, 30, 20]

king_late_game_piece_square_table = [
-50,-40,-30,-20,-20,-30,-40,-50,
-30,-20,-10,  0,  0,-10,-20,-30,
-30,-10, 20, 30, 30, 20,-10,-30,
-30,-10, 30, 40, 40, 30,-10,-30,
-30,-10, 30, 40, 40, 30,-10,-30,
-30,-10, 20, 30, 30, 20,-10,-30,
-30,-30,  0,  0,  0,  0,-30,-30,
-50,-30,-30,-30,-30,-30,-30,-50]


def calculate_piece_value(piece_type):
    if(piece_type == 1):
        return pawnValue
    elif(piece_type == 2):
        return knightValue
    elif(piece_type == 3):
        return bishopValue
    elif(piece_type == 4):
        return rookValue
    elif(piece_type == 5):
        return queenValue
    elif(piece_type == 6):
        return kingValue


def calculate_piece_value_score(board):
    pieceScore = 0
    for piece in board.piece_map():
       boardPiece = board.piece_at(piece)
       
       if(boardPiece.color == True):
           pieceScore += calculate_piece_value(boardPiece.piece_type)
       elif(boardPiece.color == False):
           pieceScore -= calculate_piece_value(boardPiece.piece_type)
    return pieceScore


def calculate_piece_value_score_for_game_stage(board):
    pieceScore = 0
    for piece in board.piece_map():
       boardPiece = board.piece_at(piece)
       if(str(boardPiece) == "k" or str(boardPiece) == "K"):
           continue
       else:
           pieceScore += calculate_piece_value(boardPiece.piece_type)
    return pieceScore

def calculate_late_game_stage(board):
    if(calculate_piece_value_score_for_game_stage(board) < 2600):
        return True
    else:
        return False
        
def generic_piece_square_value_calculator(piece, piece_square_table):
    if(piece < 8):
        return piece_square_table[56 + piece]
    elif(piece < 16):
        return piece_square_table[63 - piece]
    elif(piece < 24):
        return piece_square_table[piece + 24]
    elif(piece < 32):
        return piece_square_table[63 - piece]
    elif(piece < 40):
        return piece_square_table[piece - 8]
    elif(piece < 48):
        return piece_square_table[63 - piece]
    elif(piece < 56):
        return piece_square_table[piece - 40]
    elif(piece < 64):
        return piece_square_table[63 - piece]
    

def calculate_piece_square_table(board):
    piece_square_value = 0
    for piece in board.piece_map(): 
        if(str(board.piece_at(piece)) == "P"):
            piece_square_value += generic_piece_square_value_calculator(piece, pawn_piece_square_table)
        elif(str(board.piece_at(piece)) == "N"):
            piece_square_value += generic_piece_square_value_calculator(piece, knight_piece_square_table)
        elif(str(board.piece_at(piece)) == "B"):
            piece_square_value += generic_piece_square_value_calculator(piece, bishop_piece_square_table)
        elif(str(board.piece_at(piece)) == "R"):
            piece_square_value += generic_piece_square_value_calculator(piece, rook_piece_square_table)
        elif(str(board.piece_at(piece)) == "Q"):
            piece_square_value += generic_piece_square_value_calculator(piece, queen_piece_square_table)
        elif(str(board.piece_at(piece)) == "K"):
            if(calculate_late_game_stage(board)):
                piece_square_value += generic_piece_square_value_calculator(piece, king_late_game_piece_square_table)
            else:
                piece_square_value += generic_piece_square_value_calculator(piece, king_early_game_piece_square_table)
        if(str(board.piece_at(piece)) == "p"):
            piece_square_value -= generic_piece_square_value_calculator(piece, pawn_piece_square_table)
        elif(str(board.piece_at(piece)) == "n"):
            piece_square_value -= generic_piece_square_value_calculator(piece, knight_piece_square_table)
        elif(str(board.piece_at(piece)) == "b"):
            piece_square_value -= generic_piece_square_value_calculator(piece, bishop_piece_square_table)
        elif(str(board.piece_at(piece)) == "r"):
            piece_square_value -= generic_piece_square_value_calculator(piece, rook_piece_square_table)
        elif(str(board.piece_at(piece)) == "q"):
            piece_square_value -= generic_piece_square_value_calculator(piece, queen_piece_square_table)
        elif(str(board.piece_at(piece)) == "k"):
            if(calculate_late_game_stage(board)):
                piece_square_value -= generic_piece_square_value_calculator(piece, king_late_game_piece_square_table)
            else:
                piece_square_value -= generic_piece_square_value_calculator(piece, king_early_game_piece_square_table)
    return piece_square_value


def find_black_pawns(board):
    black_pawns = []
    for piece_index in board.piece_map():
        if(str(board.piece_at(piece_index)) == "p"):
            black_pawns.append(piece_index)
    return black_pawns

def find_white_pawns(board):
    white_pawns = []
    for piece_index in board.piece_map():
        if(str(board.piece_at(piece_index)) == "P"):
            white_pawns.append(piece_index)
    return white_pawns
    
def isolated_white_pawns(board):
    white_pawns = find_white_pawns(board) 
    non_isolated_white_pawns = 0
    for piece_index in board.piece_map():
        if(str(board.piece_at(piece_index)) == "P"):
            if(piece_index % 8 == 7):
                for piece in white_pawns:
                    if(piece % 8 == 6):
                        non_isolated_white_pawns += 1
                        break
            elif(piece_index % 8 == 6):
                for piece in white_pawns:
                    if(piece % 8 == 5 or piece % 8 == 7):
                        non_isolated_white_pawns += 1
                        break
            elif(piece_index % 8 == 5):
                for piece in white_pawns:
                    if(piece % 8 == 4 or piece % 8 == 6):
                        non_isolated_white_pawns += 1
                        break
            elif(piece_index % 8 == 4):
                for piece in white_pawns:
                    if(piece % 8 == 3 or piece % 8 == 5):
                        non_isolated_white_pawns += 1
                        break
            elif(piece_index % 8 == 3):
                for piece in white_pawns:
                    if(piece % 8 == 2 or piece % 8 == 4):
                        non_isolated_white_pawns += 1
                        break
            elif(piece_index % 8 == 2):
                for piece in white_pawns:
                    if(piece % 8 == 1 or piece % 8 == 3):
                        non_isolated_white_pawns += 1
                        break
            elif(piece_index % 8 == 1):
                for piece in white_pawns:
                    if(piece % 8 == 0 or piece % 8 == 2):
                        non_isolated_white_pawns += 1
                        break
            elif(piece_index % 8 == 0):
                for piece in white_pawns:
                    if(piece % 8 == 1):
                        non_isolated_white_pawns += 1
                        break

    return len(white_pawns) - non_isolated_white_pawns

def isolated_black_pawns(board):
    black_pawns = find_black_pawns(board) 
    non_isolated_black_pawns = 0
    for piece_index in board.piece_map():
        if(str(board.piece_at(piece_index)) == "p"):
            if(piece_index % 8 == 7):
                for piece in black_pawns:
                    if(piece % 8 == 6):
                        non_isolated_black_pawns += 1
                        break
            elif(piece_index % 8 == 6):
                for piece in black_pawns:
                    if(piece % 8 == 5 or piece % 8 == 7):
                        non_isolated_black_pawns += 1
                        break
            elif(piece_index % 8 == 5):
                for piece in black_pawns:
                    if(piece % 8 == 4 or piece % 8 == 6):
                        non_isolated_black_pawns += 1
                        break
            elif(piece_index % 8 == 4):
                for piece in black_pawns:
                    if(piece % 8 == 3 or piece % 8 == 5):
                        non_isolated_black_pawns += 1
                        break
            elif(piece_index % 8 == 3):
                for piece in black_pawns:
                    if(piece % 8 == 2 or piece % 8 == 4):
                        non_isolated_black_pawns += 1
                        break
            elif(piece_index % 8 == 2):
                for piece in black_pawns:
                    if(piece % 8 == 1 or piece % 8 == 3):
                        non_isolated_black_pawns += 1
                        break
            elif(piece_index % 8 == 1):
                for piece in black_pawns:
                    if(piece % 8 == 0 or piece % 8 == 2):
                        non_isolated_black_pawns += 1
                        break
            elif(piece_index % 8 == 0):
                for piece in black_pawns:
                    if(piece % 8 == 1):
                        non_isolated_black_pawns += 1
                        break

    return len(black_pawns) - non_isolated_black_pawns

def calculate_isolated_pawn_value(board):
    return (isolated_black_pawns(board) - isolated_white_pawns(board)) * 5


def average(lst): 
    return sum(lst) / len(lst) 


def check_attacking_king(board):
    board_value = 0
    for i in board.piece_map():
        if(str(board.piece_at(i)) == "k"):
            board_value += len(list(board.attackers(True, i))) * 200
        elif(str(board.piece_at(i)) == "K"):
            board_value -= len(board.attackers(False, i)) * 200
    return board_value


def check_checkmate(board):
    board_value = 0
    if(board.turn):
        if(board.is_checkmate()):
            board_value -= 999999999
    else:
        if(board.is_checkmate()):
            board_value += 999999999
    return board_value


def calculate_mobility_value(board):
    white_legal_move_value = 0
    black_legal_move_value = 0
    if(not board.turn):
        white_legal_move_value = len(list(board.legal_moves)) * 5
        board.turn = not board.turn
        black_legal_move_value = -len(list(board.legal_moves)) * 5
        board.turn = not board.turn
    else:
        black_legal_move_value = -len(list(board.legal_moves)) * 5
        board.turn = not board.turn
        white_legal_move_value = len(list(board.legal_moves)) * 5
        board.turn = not board.turn
    return white_legal_move_value + black_legal_move_value


def find_white_rooks(board):
    white_rooks = []
    for piece in board.piece_map():
        if(str(board.piece_at(piece)) == "R"):
            white_rooks.append(piece)
    return white_rooks

def find_black_rooks(board):
    black_rooks = []
    for piece in board.piece_map():
        if(str(board.piece_at(piece)) == "r"):
            black_rooks.append(piece)
    return black_rooks

def calculate_white_rook_on_open_file_value(board):
    white_rooks = find_white_rooks(board)
    rook_on_open_file_value = 0
    for rook_index in white_rooks:
        count = 0
        while count != 8:
            if((str(board.piece_at((rook_index % 8) + (count * 8))) != "None") and (str(board.piece_at((rook_index % 8) + (count * 8))) != "R")):
                break
            count += 1
        if(count == 8):
            rook_on_open_file_value += 15
    return rook_on_open_file_value

def calculate_black_rook_on_open_file_value(board):
    black_rooks = find_black_rooks(board)
    rook_on_open_file_value = 0
    for rook_index in black_rooks:
        count = 0
        while count != 8:
            if((str(board.piece_at((rook_index % 8) + (count * 8))) != "None") and (str(board.piece_at((rook_index % 8) + (count * 8))) != "r")):
                break
            count += 1
        if(count == 8):
            rook_on_open_file_value -= 15
    return rook_on_open_file_value

def calculate_rook_on_open_field_value(board):
    return calculate_white_rook_on_open_file_value(board) + calculate_black_rook_on_open_file_value(board)


def calculate_deep_move(board, depth, is_maximizing):
    if(depth == 0):
        return calculate_piece_value_score(board) + calculate_piece_square_table(board) + check_checkmate(board) + check_attacking_king(board) + calculate_isolated_pawn_value(board) + calculate_rook_on_open_field_value(board) + calculate_mobility_value(board) 
    else:
        legal_moves = list(board.legal_moves)
        if(is_maximizing):#white - ai
            if(len(legal_moves) == 0):
                best_move_value = check_checkmate(board)
            else:
                best_move_value = -50000
            for move in legal_moves:
                board.push(move)
                move_value = calculate_deep_move(board, depth - 1, not is_maximizing)
                best_move_value = max(best_move_value, move_value)
                board.pop()
            return best_move_value
        else:#black  player
            if(len(legal_moves) == 0):
                best_move_value = check_checkmate(board)
            else:
                best_move_value = +50000
            for move in legal_moves:
                board.push(move)
                move_value = calculate_deep_move(board, depth - 1, not is_maximizing)
                best_move_value = min(best_move_value, move_value)
                board.pop()
            return best_move_value

    
def calculate_move_first(board, depth):
    legal_moves = []
    for move in board.legal_moves:
        legal_moves.append(move)   
    best_move = None
    move_value_list = []
    if(board.turn):
        for legal_move in legal_moves:
            board.push(legal_move)
            move_value = calculate_deep_move(board, depth - 1, board.turn)
            if(len(move_value_list) == 0):
                best_move = legal_move
            elif(move_value > max(move_value_list)):
                best_move = legal_move
            move_value_list.append(move_value)
            board.pop()
        if(type(best_move) == None):
            return None
        return best_move
    else:
        for legal_move in legal_moves:
            board.push(legal_move)
            move_value = calculate_deep_move(board, depth - 1, board.turn)
            if(len(move_value_list) == 0):
                best_move = legal_move
            elif(move_value < min(move_value_list)):
                best_move = legal_move
            move_value_list.append(move_value)
            board.pop()
        if(type(best_move) == None):
            return None
        return best_move

#-------------------------------------------------------

def ai_play(board):
	return str(calculate_move_first(board,3)) # Changing the 3 to integer n will make the program think n moves ahead.
