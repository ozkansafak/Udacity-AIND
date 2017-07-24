"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def get_directional_weights(location,width,height):
    x,y = location
    # (xo_idx,yo_idx) gives the location of the center position on the board. [(3,3) for a 7x7 board]
    xo_idx = (width+1)/2-1
    yo_idx = (height+1)/2-1
    # wx, and wy are normalized distance metrics from center in x and y directions respectively. 
    # wx and wy == 0.5 when the cell is on the edge in x and y directions. 
    # wx and wy == 1.0 when the cell is in the center.
    wx = 1 - abs(x - xo_idx)/((width-1)/2) * 0.5
    wy = 1 - abs(y - yo_idx)/((height-1)/2) * 0.5
    
    return wx*wy


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    #
    my_legal_moves = game.get_legal_moves(player)
    my_weight = get_directional_weights(game.get_player_location(player), game.width, game.height)
    
    return float(len(my_legal_moves)) * my_weight


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    # 
    
    my_legal_moves = game.get_legal_moves(player)
    my_weight = get_directional_weights(game.get_player_location(player), game.width, game.height)

    opponent_legal_moves = game.get_legal_moves(game.get_opponent(player))
    opponent = game.get_opponent(player)
    opponent_weight = get_directional_weights(game.get_player_location(opponent), game.width, game.height)
    
    
    return float(len(my_legal_moves)) * my_weight - float(len(opponent_legal_moves)) * opponent_weight


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    my_legal_moves = game.get_legal_moves(player)
    opponent_legal_moves = game.get_legal_moves(game.get_opponent(player))
    
    my_score = sum([get_directional_weights(move, game.width, game.height) for move in my_legal_moves])
    opponent_score = sum([get_directional_weights(move, game.width, game.height) for move in opponent_legal_moves])
    
    return my_score - opponent_score


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        
        legal_moves = game.get_legal_moves()
        best_move = legal_moves[0] if legal_moves else (-1,-1)
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)
        except SearchTimeout:
            # Return the best move from the last completed search iteration
            pass

        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
  
        _, best_move = self.maximizing_player(game, depth)
        
        return best_move
        

    def maximizing_player(self, game, depth, best_score=float('-inf')):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if depth == 0:
            return self.score(game, self), (-2, -4)

        # get the legal_moves of the currently active maximizing_player
        legal_moves = game.get_legal_moves()
        best_move = legal_moves[0] if legal_moves else (-2,-2)

        for move in legal_moves:
            forecast_score, _ = self.minimizing_player(game.forecast_move(move), depth-1)
            if forecast_score > best_score:
                best_score, best_move = forecast_score, move

        return best_score, best_move


    def minimizing_player(self, game, depth, best_score=float('inf')):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self), (-2, -5)
        
        # get the legal_moves of the currently active minimizing player
        legal_moves = game.get_legal_moves()
        best_move = legal_moves[0] if legal_moves else (-2,-3)

        for move in legal_moves:
            forecast_score, _ = self.maximizing_player(game.forecast_move(move), depth-1)
            if forecast_score < best_score:
                best_score, best_move = forecast_score, move

        return best_score, best_move
        
        
class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-10, -10)
        
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            for depth in range(1,10**6):
                best_move = self.alphabeta(game, depth)
                
        except SearchTimeout:
            # Return the best move from the last completed search iteration
            pass

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        _, best_move = self.ab_maximizing_player(game, depth, alpha, beta)
        
        return best_move


    def ab_maximizing_player(self, game, depth, alpha, beta, best_score=float("-inf")):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if depth == 0:
            return self.score(game, self), (-1, -2)
        
        # get the legal_moves of the currently active maximizing_player
        legal_moves = game.get_legal_moves()
        best_move = legal_moves[0] if legal_moves else (-1,-4)
        
        for move in legal_moves:
            forecast_score, _ = self.ab_minimizing_player(game.forecast_move(move), depth-1, alpha, beta)

            if forecast_score > best_score:
                best_score, best_move = forecast_score, move
                
            # Prune the branch
            if best_score >= beta:
                return best_score, best_move
            
            # update alpha
            alpha = max(alpha, best_score)
        
        return best_score, best_move


    def ab_minimizing_player(self, game, depth, alpha, beta, best_score=float("inf")):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
    
        if depth == 0:
            return self.score(game, self), (-1, -3)
        
        # get the legal_moves of the currently active minimizing player
        legal_moves = game.get_legal_moves()
        best_move = legal_moves[0] if legal_moves else (-1,-5)
        
        for move in legal_moves:
            forecast_score, _ = self.ab_maximizing_player(game.forecast_move(move), depth-1, alpha, beta)
            if forecast_score < best_score:
                best_score, best_move = forecast_score, move
            
            # Prune the branch
            if best_score <= alpha:
                return best_score, best_move

            # update beta
            beta = min(beta, best_score)
                
        
        return best_score, best_move








