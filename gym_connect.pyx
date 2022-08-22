# cython: profile=True
from __future__ import division
import numpy as np
from cython.view cimport array as cvarray


cdef class gym_connect():
    cdef int rows, columns, inarow, turn
    cdef public agent, observation_space, npboard
    cdef short[:,:,:] board
    cdef short[:,] rowpercolumn,cfree
    cdef int[4][2] directions
    def __init__(self, config):
        self.rows = config['rows']
        self.columns = config['columns']
        self.inarow = config['inarow']
        self.agent = config['agent']
        self.directions = [(0,1),(1,0),(1,1),(1,-1)]
        if self.agent == None :
            self.agent = self.randomplay

        elif self.agent == 'rule':
            self.agent = self.ruleplay
#        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.npboard= np.zeros((2,self.rows,self.columns),dtype = np.short)         
        self.board = self.npboard
        self.rowpercolumn = np.zeros(self.columns,dtype = np.short)
        self.cfree= np.zeros(self.columns,dtype = np.dtype(np.short))
        self.turn = 0
        self.observation_space = (2,self.rows,self.columns)


    def _get_obs(self):
        return self.npboard

    def _get_info(self):
        return ''

    def reset(self, seed=None, return_info=False, options=None):
        #super().reset()
        self.board[:] = 0
        self.rowpercolumn[:] = 0
        self.turn = np.random.randint(0,2)
        info = ''
        if self.turn == 1:
            _, reward, done, info = self.playerstep(1,self.randomplay(self.board))
        return self.npboard.copy()

    cdef int checkalign(self, short[:,:] board, int row, int column, int n):
        cdef int x,y
        x,y = row, column
        cdef int rows,columns
        cdef int dx, dy
        cdef int i
        rows, columns = self.rows, self.columns
        for direction in range(4):
            x,y = row,column
            dx, dy = self.directions[direction]
            i = 1
            while 1:
                x += dx
                y += dy
                if x>=rows or y>=columns or x < 0 or y < 0:
                    break
                elif board[x,y]==0:
                    break
                i += 1
            x,y = row,column
            while 1:
                x -= dx
                y -= dy
                if x>=rows or y>=columns or x < 0 or y < 0:
                    break
                elif board[x,y]==0:
                    break
                i += 1
            if i>= n:
                return 1
        return 0

    cpdef int opponentplay(self):
        return self.agent(self.npboard[::-1,].copy())

    cpdef int randomplay(self, short[:,:,:] board):
        cdef int i=0
        cdef int nfree = 0
        cdef int r = 0
        for i in range(self.columns):
            if board[0,0,i]+board[1,0,i]==0:
                self.cfree[nfree]=i
                nfree+=1
        if nfree>0:
            r = np.random.randint(0,nfree)
        else:
            return -1
        #print(nfree)
        return self.cfree[r]

    cpdef int ruleplay(self, short[:,:,:] board):
        cdef int i=0
        cdef int nfree = 0
        cdef int r = 0
        cdef int lvalue = 0
        cdef int value
        cdef int row
        for i in range(self.columns):
            row = -1
            while row<self.rows-1 and board[0,row+1,i]+board[1,row+1,i]==0:
                row+=1
            if row==-1: # colonne pleine
                continue
            nboard = board[0].copy()
            nboard[row,i] = 1
            if self.checkalign(nboard, row, i, self.inarow):
                value = 2
            elif lvalue<2:
                oboard = board[1].copy()
                oboard[row,i] = 1
                if self.checkalign(oboard, row, i, self.inarow):
                    value=1
                else:
                    value = 0
            else:
                continue

            if value>lvalue:
                nfree=1
                lvalue = value
                self.cfree[0]=i
            elif value==lvalue:
                self.cfree[nfree]=i
                nfree+=1
            else:
                continue

        if nfree>0:
            r = np.random.randint(0,nfree)
        else:
            return -1
        #print(nfree)
        return self.cfree[r]



    def playerstep(self, int pid, int action):
        info = ""
        cdef int full
        cdef int i, reward, done
        if self.rowpercolumn[action] == self.rows:
            info = "Wrong move"
            reward = -2
            done = 1
            return self.npboard.copy(), reward, done, info
        else:
            self.board[pid,-self.rowpercolumn[action]-1,action] = 1
            win = self.checkalign(self.board[pid],self.rows-self.rowpercolumn[action]-1,action, self.inarow)
            full = 1
            self.rowpercolumn[action] += 1
            for i in range(self.columns):
                if self.rowpercolumn[i] != self.rows:
                    full = 0
                    break
            if win:
                return self.npboard.copy(), 1,True,''
            elif full:
                return self.npboard.copy(), 0,True,'full'
            else:
                return self.npboard.copy(), 0,False,''



    def step(self, action):
        nboard , reward, done, info = self.playerstep(0,action)
        if done : 
            return nboard, reward, done, info

        action = self.opponentplay()
        nboard, reward1, done, info1 = self.playerstep(1,action)
        if reward1 == 1:
            reward = -1
        return  self.npboard.copy(), reward, done, info+info1

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
