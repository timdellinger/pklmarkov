def sim_traditional(p_win_A, p_win_B, score_to_win, n_rallys=100, track_game_length=False, win_by_one=False, show_franction_finished=False ):
  """
  here's the docstring
  traditional scoring not rally scoring
  """


  # Error checking for input arguments
  if not isinstance(score_to_win, int) or score_to_win <= 0:
    raise ValueError("score_to_win must be an integer greater than 0")
  if not isinstance(n_rallys, int) or n_rallys <= 0:
    raise ValueError("n_rallys must be an integer greater than 0")
  if not 0 <= p_win_A <= 1:
    raise ValueError("p_win_A must be between 0 and 1")
  if not 0 <= p_win_B <= 1:
    raise ValueError("p_win_B must be between 0 and 1")

  # four possible servers, and 0 can be a score so e.g. 12 possible scores
  #  if the score_to_win is 11, which is why we add one

  number_of_game_states = 4 * (score_to_win + 1) ** 2

  transition_matrix = np.zeros((number_of_game_states,number_of_game_states))


  # enumerating the list of game states
  #
  # the simulation function doesn't rely on this particular structure,
  # so it can be changed arbitrarily without breaking the rest of the code
  #
  # for a game to 11 (i.e. 12 score states)
  # the overall formula for position in the matrix is:
  # (A_score + 48 * B_score) for A1 serving
  #   + 12 for A2 serving
  #   + 24 for B1 serving
  #   + 36 for B2 serving


  # find position in the matrix from score and server identity
  # server runs 0 -> 3 for A1, A2, B1, B2

  def find_state(A_score, B_score, server):
    return (A_score + B_score * (score_to_win+1)*4 + server * (score_to_win+1))

  # note that there are four places where the score is 11-11 (which can't happen)
  # I'll use those to respresent the "winning the game by two points" state, e.g. 12-10

  # Note that I'm keeping the score as (A score, B score, server) i.e. "A score" is always first here
  # which is different than the way that a ref calls the score

  # useful for debugging, else keep commented out
  #
  # def find_score_server(state, score_to_win):
  #     B_score = state // ((score_to_win + 1) * 4)
  #     remaining = state % ((score_to_win + 1) * 4)
  #     server = remaining // (score_to_win + 1)
  #     A_score = remaining % (score_to_win + 1)
  #      return A_score, B_score, server


  # fill in the transition matrix

  for A_score in range(score_to_win): # e.g. 0->10
    for B_score in range(score_to_win):

      # A1 wins/loses a rally
      transition_matrix[(find_state(A_score,B_score,0),find_state(A_score+1,B_score,0))]= p_win_A
      transition_matrix[(find_state(A_score,B_score,0),find_state(A_score,B_score,1))]= (1-p_win_A)

      # A2 wins/loses a rally
      transition_matrix[(find_state(A_score,B_score,1),find_state(A_score+1,B_score,1))]= p_win_A
      transition_matrix[(find_state(A_score,B_score,1),find_state(A_score,B_score,2))]= (1-p_win_A)

      # B1 wins/loses a rally
      transition_matrix[(find_state(A_score,B_score,2),find_state(A_score,B_score+1,2))]= p_win_B
      transition_matrix[(find_state(A_score,B_score,2),find_state(A_score,B_score,3))]= (1-p_win_B)

      # B2 wins/loses a rally
      transition_matrix[(find_state(A_score,B_score,3),find_state(A_score,B_score+1,3))]= p_win_B
      transition_matrix[(find_state(A_score,B_score,3),find_state(A_score,B_score,0))]= (1-p_win_B)

  # set win states (at e.g. 11 points) to a value of "1" (stationary)

  for the_server in range(2): # i.e. server 0 & 1 on team A
      for score_B in range(score_to_win-1): # i.e. only up to a score of e.g. 9 in games to 11
          transition_matrix[(find_state(score_to_win,score_B,the_server),find_state(score_to_win,score_B,the_server))]= 1
      # skip the score of e.g. 10, now catch the score of e.g 11-11:
      transition_matrix[(find_state(score_to_win,score_to_win,the_server),find_state(score_to_win,score_to_win,the_server))]= 1

  for the_server in range(2,3): # i.e. server 2 & 3 on team B
    for score_A in range(score_to_win-1): # also only up to a score of e.g. 9
        transition_matrix[(find_state(score_A,score_to_win,the_server),find_state(score_A,score_to_win,the_server))]= 1
    # skip the score of e.g. 10, now catch the score of e.g 11-11:
    transition_matrix[(find_state(score_to_win,score_to_win,the_server),find_state(score_to_win,score_to_win,the_server))]= 1

  # cases in 'win by 2' scenarios
  # recall that the (11,11,_) states are used here for a win,
  # and (10,10,_) represents all tie scores e.g. 13-13
  # and e.g. the (10,11,_) states represent all advantage scores e.g. 14-15-2

  # team A wins the game from 11-10
  transition_matrix[ (find_state(score_to_win,score_to_win-1,0),find_state(score_to_win,score_to_win,0))] = p_win_A
  transition_matrix[ (find_state(score_to_win,score_to_win-1,1),find_state(score_to_win,score_to_win,1))] = p_win_A

  # team A loses a rally at 11-10
  transition_matrix[ (find_state(score_to_win,score_to_win-1,0),find_state(score_to_win,score_to_win-1,1))] = (1-p_win_A)
  transition_matrix[ (find_state(score_to_win,score_to_win-1,1),find_state(score_to_win,score_to_win-1,2))] = (1-p_win_A)

  # team A wins a rally at 10-11, game reverts to the 10-10 state (tie game in win by 2 territory)
  transition_matrix[ (find_state(score_to_win-1,score_to_win,0),find_state(score_to_win-1,score_to_win-1,0))] = p_win_A
  transition_matrix[ (find_state(score_to_win-1,score_to_win,1),find_state(score_to_win-1,score_to_win-1,1))] = p_win_A

  ## team A loses a rally at 10-11
  transition_matrix[ (find_state(score_to_win-1,score_to_win,0),find_state(score_to_win-1,score_to_win,1))] = (1-p_win_A)
  transition_matrix[ (find_state(score_to_win-1,score_to_win,1),find_state(score_to_win-1,score_to_win,2))] = (1-p_win_A)

  # team B wins the game from 10-11
  transition_matrix[ (find_state(score_to_win-1,score_to_win,2),find_state(score_to_win,score_to_win,2))] = p_win_B
  transition_matrix[ (find_state(score_to_win-1,score_to_win,3),find_state(score_to_win,score_to_win,3))] = p_win_B

  # team B loses a rally at 10-11
  transition_matrix[ (find_state(score_to_win-1,score_to_win,2),find_state(score_to_win-1,score_to_win,3))] = (1-p_win_B)
  transition_matrix[ (find_state(score_to_win-1,score_to_win,3),find_state(score_to_win-1,score_to_win,0))] = (1-p_win_B)

  # team B wins a rally at 11-10, game reverts to the 10-10 state (tie game in win by 2 territory)
  transition_matrix[ (find_state(score_to_win,score_to_win-1,2),find_state(score_to_win-1,score_to_win-1,2))] = p_win_B
  transition_matrix[ (find_state(score_to_win,score_to_win-1,3),find_state(score_to_win-1,score_to_win-1,3))] = p_win_B

  # team B loses a rally at 11-10
  transition_matrix[ (find_state(score_to_win,score_to_win-1,2),find_state(score_to_win,score_to_win-1,3))] = (1-p_win_B)
  transition_matrix[ (find_state(score_to_win,score_to_win-1,3),find_state(score_to_win,score_to_win-1,0))] = (1-p_win_B)

  # code useful for debugging / verifivacation:

  # print(transition_matrix)

  # making sure things sum correctly:
  # print(f"number of rows that sum to 1 (should be {number_of_game_states}):")
  # print(list(transition_matrix.sum(axis=1)).count(1.))
  # print("number of empty colums (should be 0):")
  # print(list(transition_matrix.sum(axis=0)).count(0))

  fraction_complete = [0]

  # there might be a clever way to get to this using floor() and the transition matrix,
  # but I'll just build it up
  #
  # I want a "1" in every "game over" state
  fracion_complete_counting_tool = np.zeros(number_of_game_states)
  for the_server in range(4):
    for losing_score in range (score_to_win-1):
      fracion_complete_counting_tool[find_state(score_to_win,losing_score,the_server)] = 1
      fracion_complete_counting_tool[find_state(losing_score,score_to_win,the_server)] = 1

  fracion_complete_counting_tool[find_state(score_to_win,score_to_win,0)] = 1
  fracion_complete_counting_tool[find_state(score_to_win,score_to_win,1)] = 1
  fracion_complete_counting_tool[find_state(score_to_win,score_to_win,2)] = 1
  fracion_complete_counting_tool[find_state(score_to_win,score_to_win,3)] = 1

  # initializing
  outcomeMatrix = transition_matrix

  # code useful when debugging / verifying
  # print ( [(i, find_score_server(index,score_to_win)) for index, i in enumerate(outcomeMatrix[find_state(0,0,1),:]) if i != 0 ] )
  # print ( [(i, find_score_server(index,score_to_win)) for index, i in enumerate(outcomeMatrix[find_state(score_to_win-1,score_to_win-1,1),:]) if i != 0 ] )


  # play out the rallys
  for rally_number in range(n_rallys-1):
    outcomeMatrix = np.matmul(transition_matrix,outcomeMatrix)
    fraction_complete_so_far = np.matmul(fracion_complete_counting_tool,outcomeMatrix[find_state(0,0,1),:])
    fraction_complete.append(fraction_complete_so_far)
    # useful for debugging / verification
    # print(f"rally {rally_number} complete")
    # print ( [(i, find_score_server(index,score_to_win)) for index, i in enumerate(outcomeMatrix[find_state(0,0,1),:]) if i != 0 ] )
    # print ( [(i, find_score_server(index,score_to_win)) for index, i in enumerate(outcomeMatrix[find_state(score_to_win-1,score_to_win-1,1),:]) if i != 0 ] )
    # print ( outcomeMatrix[:,find_state(0,0,1)] )


  # building up more tools
  a_win_counter = np.zeros(number_of_game_states)
  b_win_counter = np.zeros(number_of_game_states)
  for the_server in range(4):
    for losing_score in range (score_to_win-1):
      a_win_counter[find_state(score_to_win,losing_score,the_server)] = 1
      b_win_counter[find_state(losing_score,score_to_win,the_server)] = 1

  a_win_counter[find_state(score_to_win,score_to_win,0)] = 1
  a_win_counter[find_state(score_to_win,score_to_win,1)] = 1
  b_win_counter[find_state(score_to_win,score_to_win,2)] = 1
  b_win_counter[find_state(score_to_win,score_to_win,3)] = 1

  a_wins = np.matmul(a_win_counter,outcomeMatrix[find_state(0,0,1),:])
  b_wins = np.matmul(b_win_counter,outcomeMatrix[find_state(0,0,1),:])

  differential_fraction_complete = [fraction_complete[0]] + [
      fraction_complete[i] - fraction_complete[i - 1]
      for i in range(1, len(fraction_complete))
      ]
  return a_wins, b_wins, a_wins + b_wins, differential_fraction_complete

