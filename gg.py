import pygame, random, heapq
from enum import Enum
from collections import deque, namedtuple

# Config
CELL, W, H, FPS = 20, 30, 20, 30
Point = namedtuple('Point', 'x y')

class Dir(Enum):
    UP=(0,-1); DOWN=(0,1); LEFT=(-1,0); RIGHT=(1,0)

# Game
class SnakeGame:
    def __init__(self):
        self.reset()
    def reset(self):
        mid = Point(W//2, H//2)
        self.snake = deque([mid]); self.direction = random.choice(list(Dir))
        self.length = 3; self.spawn_food()
    def spawn_food(self):
        while True:
            p = Point(random.randrange(W), random.randrange(H))
            if p not in self.snake:
                self.food = p; return
    def step(self, d: Dir):
        head = self.snake[0]
        nxt = Point(head.x + d.value[0], head.y + d.value[1])
        if not (0 <= nxt.x < W and 0 <= nxt.y < H) or nxt in self.snake:
            return False
        self.snake.appendleft(nxt)
        if nxt == self.food:
            self.length += 1; self.spawn_food()
        else:
            while len(self.snake) > self.length:
                self.snake.pop()
        self.direction = d
        return True

def manhattan(a, b): return abs(a.x - b.x) + abs(a.y - b.y)
def neighbors(pt):
    for d in Dir: yield Point(pt.x + d.value[0], pt.y + d.value[1]), d

def greedy_move(game):
    head = game.snake[0]
    options = [(manhattan(pt, game.food), d) 
               for pt, d in neighbors(head)
               if 0 <= pt.x < W and 0 <= pt.y < H and pt not in game.snake]
    return min(options, key=lambda x: x[0])[1] if options else game.direction

# Fixed A* with tie-breaking counter
def astar(game, limit=5000):
    start, goal = game.snake[0], game.food
    openq, closed, counter = [], set(), 0
    heapq.heappush(openq, (manhattan(start, goal), 0, counter, start, []))
    while openq:
        f, g, _, cur, path = heapq.heappop(openq)
        if g > limit or (cur, g) in closed: continue
        closed.add((cur, g))
        if cur == goal:
            return path[0] if path else game.direction
        for nxt, d in neighbors(cur):
            if not (0 <= nxt.x < W and 0 <= nxt.y < H): continue
            if nxt in game.snake and nxt != game.snake[-1]: continue
            counter += 1
            heapq.heappush(openq, (g+1+manhattan(nxt,goal), g+1, counter, nxt, path+[d]))
    return None

# Tail reachability check
def can_reach_tail(snake, tail):
    seen, dq, body = {snake[0]}, deque([snake[0]]), set(snake)
    while dq:
        cur = dq.popleft()
        if cur == tail: return True
        for nxt, _ in neighbors(cur):
            if 0 <= nxt.x < W and 0 <= nxt.y < H and nxt not in seen and nxt not in body:
                seen.add(nxt); dq.append(nxt)
    return False

# A* with forward-checking
def astar_forward(game):
    d = astar(game)
    if d is None: return greedy_move(game)
    head = game.snake[0]; nxt_head = Point(head.x + d.value[0], head.y + d.value[1])
    future = deque(game.snake); future.appendleft(nxt_head)
    if len(future) > game.length: future.pop()
    return d if can_reach_tail(future, future[-1]) else greedy_move(game)

# Main loop
def main(mode='forward'):
    pygame.init()
    screen = pygame.display.set_mode((W * CELL, H * CELL))
    clock = pygame.time.Clock()
    game = SnakeGame()
    print("Modes: 'greedy', 'astar', 'forward'")
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
        if mode == 'greedy':
            d = greedy_move(game)
        elif mode == 'astar':
            d = astar(game) or greedy_move(game)
        elif mode == 'forward':
            d = astar_forward(game)
        else:
            d = game.direction
        if d is None:
            d = game.direction
        alive = game.step(d)
        if not alive:
            print("Game Over. Score:", game.length - 3)
            game.reset()
        screen.fill((30, 30, 30))
        for p in game.snake:
            pygame.draw.rect(screen, (0, 200, 0), (p.x * CELL, p.y * CELL, CELL, CELL))
        pygame.draw.rect(screen, (200, 0, 0), (game.food.x * CELL, game.food.y * CELL, CELL, CELL))
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    # main(mode='greedy')
    # main(mode='forward')
    main(mode='astar')