import numpy as np


class gym_connect():
#    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config):
        self.rows = config['rows']
        self.columns = config['columns']
        self.inarow = config['inarow']
        self.agent = config['agent']
        if self.agent == None :
            self.agent = self.randomplay
#        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.board= np.zeros((2,self.rows,self.columns),dtype = np.int8)         
        self.rowpercolumn = np.zeros(self.columns,dtype = np.int8)
        self.turn = 0


        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.board

    def _get_info(self):
        return ''

    def reset(self, seed=None, return_info=False, options=None):
        #super().reset()
        self.board = np.zeros((2,self.rows,self.columns),dtype = np.int8)         
        observation = self._get_obs()
        self.rowpercolumn = np.zeros(self.columns,dtype = np.int8)
        self.turn = np.random.randint(0,2)
        info = ''
        if self.turn == 1:
            reward, done, info = self.playerstep(1,self.opponentplay())
        return (observation, info) if return_info else observation
    
    def checkalign(self, board, row, column, n):
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        for direction in directions:
            cell = [row,column]
            i = 1
            while 1:
                cell[0] += direction[0]
                cell[1] += direction[1]
                if cell[0]>=self.rows or cell[1]>=self.columns or cell[0] < 0 or cell[1] < 0:
                    break
                elif board[cell[0],cell[1]]==0:
                    break
                i += 1
            cell = [row,column]
            while 1:
                cell[0] -= direction[0]
                cell[1] -= direction[1]
                if cell[0]>=self.rows or cell[1]>=self.columns or cell[0] < 0 or cell[1] < 0:
                    break
                elif board[cell[0],cell[1]]==0:
                    break
                i += 1
            if i>= n:
                return True
        return False

    def opponentplay(self):
        board = self.board[[1,0]]
        return self.agent(board)
                    
    def randomplay(self, board):
        i=0
        freecolumns = []
        for i in range(self.columns):
            if board[0,0,i]+board[1,0,i]==0:
                freecolumns.append(i)
        return freecolumns[np.random.randint(0,len(freecolumns))]

                
    def playerstep(self, pid, action):
        info = ""
        if self.rowpercolumn[action] == self.rows:
            info = "Wrong move"
            reward = -2
            done = True
            return reward, done, info
        else:
            self.board[pid,-self.rowpercolumn[action]-1,action] = 1
            win = self.checkalign(self.board[pid],self.rows-self.rowpercolumn[action]-1,action, self.inarow)
            full = np.sum(self.board)==self.rows*self.columns
            self.rowpercolumn[action] += 1
            if win:
                return 1,True,''
            elif full:
                return 0,True,'full'
            else:
                return 0,False,''



    def step(self, action, agent=None):
        reward, done, info = self.playerstep(0,action)
        if done : 
            return self.board.copy(), reward, done, info

        action = self.opponentplay()
        reward1, done, info1 = self.playerstep(1,action)
        if reward1 == 1:
            reward = -1
        return  self.board.copy(), reward, done, info+info1

"""
    def render(self, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

"""
#c = ConnectX()
#print(c.reset())
#done = False
#while not done:
#    board, reward, done, info = c.step(c.randomplay())
#    print(board, reward, done, info)
