import numpy as np

def three_in_a_row(board):
    """ Returns true if 3 in a row, but no diagonals

    :param board:
    :return:
    """
    three_in_a_row = False
    for row in board:
        three_in_a_row = three_in_a_row or np.all(row)
    for column in board:
        three_in_a_row = three_in_a_row or np.all(column)

    return three_in_a_row


if __name__ == '__main__':

    board = np.zeros((3, 3))
    player_one_wins = False
    player_two_wins = False
    state_table = numpy

    while not player_one_wins and not player_two_wins:

        print(board)


        row = int(input("Which row?"))
        column = int(input("Which column?"))

        board[row, column] = 1.0

        player_one_wins = three_in_a_row(board == 1.0)
        player_two_wins = three_in_a_row(board == 2.0)

