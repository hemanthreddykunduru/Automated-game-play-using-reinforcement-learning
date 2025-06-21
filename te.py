import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from enum import Enum
import matplotlib.pyplot as plt
import os

CELL, W, H, FPS = 20, 30, 20, 15
Point = namedtuple('Point', 'x y')

class Dir(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class SnakeGameRL:
    def __init__(self):
        self.reset()
        
    def reset(self):
        mid = Point(W//2, H//2)
        self.snake = deque([mid])
        self.direction = Dir.RIGHT
        self.length = 3
        self.spawn_food()
        self.frame_iteration = 0
        
    def spawn_food(self):
        while True:
            p = Point(random.randrange(W), random.randrange(H))
            if p not in self.snake:
                self.food = p
                return
                
    def step(self, action):
        self.frame_iteration += 1
        self._move(action)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.length - 3
        if self.snake[0] == self.food:
            self.length += 1
            reward = 10
            self.spawn_food()
        else:
            while len(self.snake) > self.length:
                self.snake.pop()
        head = self.snake[0]
        old_distance = abs(head.x - self.food.x) + abs(head.y - self.food.y)
        
        # Small reward for moving toward food
        if len(self.snake) > 1:
            prev_head = self.snake[1]
            prev_distance = abs(prev_head.x - self.food.x) + abs(prev_head.y - self.food.y)
            if old_distance < prev_distance:
                reward += 1  # Moving closer to food
            else:
                reward -= 0.5  # Moving away from food
        reward += 0.1
        
        return reward, game_over, self.length - 3
        
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[0]
        # Hits boundary
        if pt.x < 0 or pt.x >= W or pt.y < 0 or pt.y >= H:
            return True
        # Hits itself
        if pt in list(self.snake)[1:]:
            return True
        return False
        
    def _move(self, action):
        clock_wise = [Dir.RIGHT, Dir.DOWN, Dir.LEFT, Dir.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        
        head = self.snake[0]
        x, y = self.direction.value
        new_head = Point(head.x + x, head.y + y)
        self.snake.appendleft(new_head)
        
    def get_state(self):
        head = self.snake[0]
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)
        
        dir_l = self.direction == Dir.LEFT
        dir_r = self.direction == Dir.RIGHT
        dir_u = self.direction == Dir.UP
        dir_d = self.direction == Dir.DOWN
        
        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),
            
            # Danger right
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),
            
            # Danger left
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            
            # Move direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Food location 
            self.food.x < head.x,  # food left
            self.food.x > head.x,  # food right
            self.food.y < head.y,  # food up
            self.food.y > head.y   # food down
        ]
        
        return np.array(state, dtype=int)

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class DQNTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        device = next(self.model.parameters()).device
        state = state.to(device)
        next_state = next_state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=100_000)
        self.model = DQN(11, 256, 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")
        self.trainer = DQNTrainer(self.model, lr=0.001, gamma=self.gamma)
        self.scores = []
        self.mean_scores = []
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    def get_action(self, state):
        self.epsilon = max(10, 200 - self.n_games * 0.5)
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
    def save_model(self, filename='snake_dqn_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_games': self.n_games,
        }, filename)
        print(f"Model saved as {filename}")
    def load_model(self, filename='snake_dqn_model.pth'):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.n_games = checkpoint.get('n_games', 0)
            print(f"Model loaded from {filename}")
            return True
        return False

def train():
    print("🎯 TRAINING RECOMMENDATIONS:")
    print("📊 Expected Training Progress:")
    print("   • Games 1-500: Learning basic survival (scores 0-3)")
    print("   • Games 500-1500: Learning food seeking (scores 3-10)")
    print("   • Games 1500-3000: Developing strategy (scores 10-25)")
    print("   • Games 3000+: Mastering the game (scores 25+)")
    print("\n⏱️  Estimated training time:")
    print("   • 1000 games ≈ 5-10 minutes")
    print("   • 3000 games ≈ 15-30 minutes")
    print("   • 5000+ games ≈ 30+ minutes (recommended)")
    print("\n" + "="*60)
    target_games = input("Enter target number of games to train (recommended: 5000): ").strip()
    try:
        target_games = int(target_games) if target_games else 5000
    except:
        target_games = 5000
    print(f"Training target: {target_games} games")
    print("Press Ctrl+C to stop early and save progress")
    print("="*60)
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameRL()
    milestone_games = [100, 500, 1000, 2000, 3000, 5000]
    last_100_scores = deque(maxlen=100)
    print("Starting training...")
    try:
        while agent.n_games < target_games:
            state_old = game.get_state()
            final_move = agent.get_action(state_old)
            reward, done, score = game.step(final_move)
            state_new = game.get_state()
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                last_100_scores.append(score)
                if score > record:
                    record = score
                    agent.save_model('best_snake_model.pth')
                    print(f"🏆 NEW RECORD! Game {agent.n_games}: Score {score}")
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                if agent.n_games % 100 == 0:
                    recent_avg = sum(last_100_scores) / len(last_100_scores)
                    exploration_rate = max(10, 200 - agent.n_games * 0.5)
                    print(f"Game {agent.n_games:4d} | Score: {score:2d} | Record: {record:2d} | "
                          f"Avg(100): {recent_avg:4.1f} | Exploration: {exploration_rate:3.0f}%")
                    agent.save_model(f'checkpoint_game_{agent.n_games}.pth')
                if agent.n_games in milestone_games:
                    recent_avg = sum(last_100_scores) / len(last_100_scores) if last_100_scores else 0
                    print(f"\n🎯 MILESTONE REACHED: {agent.n_games} games")
                    print(f"   Record Score: {record}")
                    print(f"   Recent Average (last 100): {recent_avg:.1f}")
                    print(f"   Overall Average: {mean_score:.1f}")
                    print("-" * 50)
    except KeyboardInterrupt:
        print(f"\n⏹️  Training stopped by user at game {agent.n_games}")
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"📊 Final Statistics:")
    print(f"   Games Played: {agent.n_games}")
    print(f"   Best Score: {record}")
    print(f"   Overall Average: {total_score/max(agent.n_games,1):.2f}")
    if last_100_scores:
        recent_avg = sum(last_100_scores) / len(last_100_scores)
        print(f"   Recent Average (last 100): {recent_avg:.2f}")
    if record >= 20:
        print("🏆 EXCELLENT! Your AI has learned to play very well!")
    elif record >= 10:
        print("👍 GOOD! Your AI is getting better. Consider training more for higher scores.")
    elif record >= 5:
        print("📈 PROGRESS! Your AI is learning. More training recommended.")
    else:
        print("⚠️  NEEDS MORE TRAINING! Try training for 3000+ games.")
    print("\n💾 Saving trained model...")
    agent.save_model('trained_snake_model.pth')
    agent.save_model('final_snake_model.pth')
    print("✅ Model saved successfully!")
    return plot_scores, plot_mean_scores
def play_with_ai(model_path=None, visualize=True):
    agent = Agent()
    if model_path is None:
        model_priority = [
            'trained_snake_model.pth',
            'best_snake_model.pth',
            'final_snake_model.pth'
        ]
        model_path = None
        for model_file in model_priority:
            if os.path.exists(model_file):
                model_path = model_file
                break
        if model_path is None:
            print("ERROR: No trained model found!")
            print("Available models should be:")
            for model_file in model_priority:
                print(f"  - {model_file}")
            print("\nPlease train the model first by running the training mode.")
            return
    if not agent.load_model(model_path):
        print(f"ERROR: Could not load model from {model_path}")
        print("Please check if the model file exists and is valid.")
        return
    print(f"Successfully loaded model: {model_path}")
    agent.model.eval()
    agent.epsilon = 0 
    if visualize:
        pygame.init()
        screen = pygame.display.set_mode((W * CELL, H * CELL))
        pygame.display.set_caption(f'Snake AI - Playing with {model_path}')
        clock = pygame.time.Clock()
    game = SnakeGameRL()
    total_games = 0
    total_score = 0
    max_score = 0
    print("AI is now playing Snake automatically!")
    print("Press Ctrl+C to stop" + (" or close the window" if visualize else ""))
    try:
        while True:
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
            state = game.get_state()
            with torch.no_grad():
                final_move = agent.get_action(state)
            reward, done, score = game.step(final_move)
            if visualize:
                screen.fill((30, 30, 30))
                for i, segment in enumerate(game.snake):
                    color = (0, 255, 0) if i == 0 else (0, 200, 0)
                    pygame.draw.rect(screen, color, 
                                   (segment.x * CELL, segment.y * CELL, CELL, CELL))
                
                pygame.draw.rect(screen, (255, 0, 0), 
                               (game.food.x * CELL, game.food.y * CELL, CELL, CELL))
                font = pygame.font.Font(None, 24)
                stats_text = [
                    f'Current Score: {score}',
                    f'Games Played: {total_games}',
                    f'Max Score: {max_score}',
                    f'Avg Score: {total_score/max(total_games,1):.1f}' if total_games > 0 else 'Avg Score: 0.0',
                    f'Model: {os.path.basename(model_path)}'
                ]
                for i, text in enumerate(stats_text):
                    rendered_text = font.render(text, True, (255, 255, 255))
                    screen.blit(rendered_text, (10, 10 + i * 25))
                pygame.display.flip()
                clock.tick(FPS)
            if done:
                total_games += 1
                total_score += score
                max_score = max(max_score, score)
                avg_score = total_score / total_games
                print(f'Game {total_games}: Score {score}, Max: {max_score}, Average: {avg_score:.2f}')
                game.reset()
    except KeyboardInterrupt:
        print(f"\nPlayback stopped by user.")
        print(f"Final Statistics:")
        print(f"  Games played: {total_games}")
        print(f"  Max score: {max_score}")
        print(f"  Average score: {total_score/max(total_games,1):.2f}")
        print(f"  Model used: {model_path}")
        if visualize:
            pygame.quit()
def main():
    print("=" * 50)
    print("🐍 Snake Game with Deep Q-Learning (GPU)")
    print("=" * 50)
    print("Choose mode:")
    print("1. 🎯 Train the AI")
    print("2. 🎮 Play automatically with trained AI (with visualization)")
    print("3. ⚡ Test AI performance (no visualization, fast)")
    print("4. 📊 Check available models")
    choice = input("\nEnter choice (1/2/3/4): ").strip()
    if choice == '1':
        print("\n🎯 Starting AI Training...")
        print("The model will be automatically saved after training.")
        train()
    elif choice == '2':
        print("\n🎮 Loading trained model for automatic play...")
        play_with_ai(visualize=True)
    elif choice == '3':
        print("\n⚡ Testing AI performance...")
        play_with_ai(visualize=False)
    elif choice == '4':
        print("\n📊 Checking available models...")
        model_files = [
            'trained_snake_model.pth',
            'best_snake_model.pth', 
            'final_snake_model.pth'
        ]
        available_models = []
        for model_file in model_files:
            if os.path.exists(model_file):
                available_models.append(model_file)
                size = os.path.getsize(model_file) / 1024  # KB
                print(f"  ✅ {model_file} ({size:.1f} KB)")
            else:
                print(f"  ❌ {model_file} (not found)")
        if not available_models:
            print("\n⚠️  No trained models found!")
            print("Please train the AI first using option 1.")
        else:
            print(f"\n✅ Found {len(available_models)} trained model(s).")
            print("You can now use option 2 or 3 to play automatically.")
    else:
        print("❌ Invalid choice! Please enter 1, 2, 3, or 4.")
if __name__ == "__main__":
    main()