import pygame
import random
import math
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim


# --- Logging setup ---
today = datetime.date.today().strftime("%Y-%m-%d")
exp_num = 5
exp_name = f"{today}_experiment_{exp_num}"
os.makedirs(exp_name, exist_ok=True)
log_file = open(os.path.join(exp_name, "log.txt"), "w")
log_file.write("success,steps\n")
log_file.flush()


# --- Parameters ---
WIDTH, HEIGHT = 2000, 1200
dt = 0.9
globalDamping = 0.995
baseFrictionHigh = 0.9
baseFrictionLow = 0.05
springK = 0.25
contractionAlpha = 0.55
contractionSpeed = 0.05
THRESHOLD = 0.2  # contraction threshold
NOISE_STD = 0.1  # Gaussian noise for exploration

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

# --- Target ---
target_radius = 80
target_x = random.randint(200, WIDTH - 200)
target_y = random.randint(200, HEIGHT - 200)

# --- Model ---
class Controller(nn.Module):
    def __init__(self, input_dim=5, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

controller = Controller()
opt = optim.Adam(controller.parameters(), lr=1e-3)

# --- Classes ---
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.radius = 6
        self.friction = baseFrictionLow
        self.channels = [random.uniform(-1, 1) for _ in range(5)]

    def accumulated_channels(self, depth=2):
        visited = set()
        acc = [0.0] * len(self.channels)
        def dfs(idx, d):
            if d > depth or idx in visited:
                return
            visited.add(idx)
            node = nodes[idx]
            for i, v in enumerate(node.channels):
                acc[i] += 1 / (1 + math.exp(-v))
            for e in edges:
                if e.a == idx: dfs(e.b, d + 1)
                elif e.b == idx: dfs(e.a, d + 1)
        dfs(nodes.index(self), 0)
        return acc

    def distance_to_target(self):
        dx = self.x - target_x
        dy = self.y - target_y
        d = math.sqrt(dx*dx + dy*dy)
        return max(0, d - target_radius)

class Edge:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.rest = dist(nodes[a], nodes[b])
        self.phase = 0
        self.contracting = False

def dist(n1, n2):
    return math.sqrt((n1.x-n2.x)**2 + (n1.y-n2.y)**2)

# --- Graph ---
nodes = []
edges = []

def buildConnectedGraph(N=20):
    global nodes, edges
    nodes = []
    edges = []
    for i in range(N):
        angle = (i/N)*math.pi*2
        r = 100 + random.random()*50
        cx = WIDTH/2 + math.cos(angle)*r
        cy = HEIGHT/2 + math.sin(angle)*r
        nodes.append(Node(cx, cy))
    for _ in range(2):
        for i in range(1, N):
            j = random.randrange(i)
            edges.append(Edge(i, j))

def contract_edge(a, b):
    for e in edges:
        if e.a == a and e.b == b:
            e.contracting = True
            break

# --- Physics ---
def update_physics():
    for e in edges:
        if e.contracting:
            e.phase += contractionSpeed
            if e.phase >= 1:
                e.phase = 1
                e.contracting = False
        else:
            e.phase -= contractionSpeed
            if e.phase < 0:
                e.phase = 0

    for e in edges:
        n1 = nodes[e.a]
        n2 = nodes[e.b]
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        d = math.sqrt(dx*dx + dy*dy)+1e-8
        L = e.rest*(1 - contractionAlpha*e.phase)
        F = springK*(d - L)
        fx, fy = F*dx/d, F*dy/d
        n1.vx += fx
        n1.vy += fy
        n2.vx -= fx
        n2.vy -= fy

    for e in edges:
        if e.phase > 0:
            base = nodes[e.a]
            lifted = nodes[e.b]
            base.friction = baseFrictionHigh*(1 - 0.5*e.phase) + baseFrictionLow*(0.5*e.phase)
            lifted.friction = baseFrictionHigh*(1 - e.phase) + baseFrictionLow*e.phase

    for n in nodes:
        n.vx *= n.friction
        n.vy *= n.friction
        n.vx *= globalDamping
        n.vy *= globalDamping
        n.x += n.vx*dt
        n.y += n.vy*dt

# --- Training ---
def training_step():
    old_dists = [n.distance_to_target() for n in nodes]

    logprobs = []
    for e in edges:
        n1, n2 = nodes[e.a], nodes[e.b]
        s1 = controller(torch.tensor(n1.accumulated_channels(), dtype=torch.float32))
        s2 = controller(torch.tensor(n2.accumulated_channels(), dtype=torch.float32))

        # add Gaussian noise for exploration
        noisy_s1 = s1 + torch.randn(1) * NOISE_STD
        noisy_s2 = s2 + torch.randn(1) * NOISE_STD

        if abs(noisy_s1.item() - noisy_s2.item()) > THRESHOLD:
            if noisy_s1.item() < noisy_s2.item():
                contract_edge(e.a, e.b)
            else:
                contract_edge(e.b, e.a)

        avg_signal = (s1 + s2) / 2
        logprobs.append(torch.log(torch.clamp(avg_signal, 1e-6, 1)))

    update_physics()

    new_dists = [n.distance_to_target() for n in nodes]
    avg_reward = (sum(old_dists) - sum(new_dists)) / len(nodes)

    if logprobs:
        loss = -(torch.stack(logprobs) * avg_reward).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return avg_reward, loss.item()
    return 0, 0


# --- Success counter ---
successes = 0

def regenerate_target():
    global target_x, target_y
    target_x = random.randint(200, WIDTH - 200)
    target_y = random.randint(200, HEIGHT - 200)

# --- Center of mass trail ---
trail = []

def update_trail():
    cx = sum(n.x for n in nodes)/len(nodes)
    cy = sum(n.y for n in nodes)/len(nodes)
    trail.append((cx, cy))
    return cx, cy


def inverse_colors_before_screenshot():
    screen.fill((255, 255, 255))
    if len(trail) > 1:
        pygame.draw.lines(screen, (10, 10, 20), False, trail, 2)


# --- Drawing ---
def draw(loss_val, reward_val, cx, cy, successes, steps_since_last, do_inverse=False):
    SCREEN_FILL = (10, 10, 20)
    TRAIL_COLOR = (255, 255, 255)
    EDGE_COLOR_1 = (200, 50, 50)  # red
    EDGE_COLOR_2 = (150, 150, 150)  # gray
    TARGET_COLOR = (255, 0, 0, 80)
    LOSS_TEXT_COLOR = (255, 255, 255)
    REWARD_TEXT_COLOR = (100, 255, 100)
    DIST_TEXT_COLOR = (255, 200, 200)
    STEPS_TEXT_COLOR = (255, 200, 200)
    SUCCESS_TEXT_COLOR = (0, 200, 255)

    if do_inverse:
        SCREEN_FILL = (245, 245, 235)
        TRAIL_COLOR = (10, 10, 20)
        TARGET_COLOR = (200, 50, 50)
        STEPS_TEXT_COLOR = (0, 0, 0)
    
    screen.fill(SCREEN_FILL)

    if len(trail) > 1:
        pygame.draw.lines(screen, TRAIL_COLOR, False, trail, 2)

    target_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.circle(target_surface, TARGET_COLOR, (target_x, target_y), target_radius)
    screen.blit(target_surface, (0, 0))
    
    for e in edges:
        n1, n2 = nodes[e.a], nodes[e.b]
        color = EDGE_COLOR_1 if e.phase>0 else EDGE_COLOR_2
        pygame.draw.line(screen, color, (n1.x, n1.y), (n2.x, n2.y), 2)

    for i, n in enumerate(nodes):
        f = (n.friction-baseFrictionLow)/(baseFrictionHigh-baseFrictionLow)
        f = max(0, min(1, f))
        r = 30*(1-f)+20*f
        g = 200*f+80*(1-f)
        b = 200*(1-f)+120*f
        pygame.draw.circle(screen, (int(r),int(g),int(b)), (int(n.x),int(n.y)), n.radius)

    pygame.draw.circle(screen, (255,255,255), (int(cx),int(cy)), 3)

    loss_text = font.render(f"Loss: {loss_val:.4f}", True, LOSS_TEXT_COLOR)
    reward_text = font.render(f"Reward: {reward_val:.4f}", True, REWARD_TEXT_COLOR)
    dist_text = font.render(f"Center→Target: {math.sqrt((cx-target_x)**2+(cy-target_y)**2):.1f}", True, DIST_TEXT_COLOR)
    steps_text = font.render(f"Steps: {steps_since_last}", True, STEPS_TEXT_COLOR)
    success_text = font.render(f"Successes: {successes}", True, SUCCESS_TEXT_COLOR)
    if do_inverse:
        screen.blit(steps_text, (20, 20))
    else:
        screen.blit(loss_text, (20, 20))
        screen.blit(reward_text, (20, 50))
        screen.blit(dist_text, (20, 80))
        screen.blit(steps_text, (20, 110))
        screen.blit(success_text, (20, 140))

    pygame.display.flip()

# --- Main loop ---
buildConnectedGraph()

running, successes, steps_since_last = True, 0, 0

while running:
    reward, loss = training_step()
    cx, cy = update_trail()
    steps_since_last += 1

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                trail.clear()
                regenerate_target()
            elif event.key == pygame.K_s:
                draw(loss, reward, cx, cy, successes, steps_since_last, do_inverse=True)
                screenshot_path = os.path.join(exp_name, f"screenshot_{successes}_{steps_since_last}.png")
                pygame.image.save(screen, screenshot_path)
        if event.type == pygame.QUIT:
            running = False

    # --- Check success ---
    if math.sqrt((cx - target_x) ** 2 + (cy - target_y) ** 2) < target_radius:
        successes += 1
        print(f"SUCCESS {successes} in {steps_since_last} steps")

        log_file.write(f"{successes},{steps_since_last}\n")
        log_file.flush()

        draw(loss, reward, cx, cy, successes, steps_since_last, do_inverse=True)
        screenshot_path = os.path.join(exp_name, f"success_{successes}.png")
        pygame.image.save(screen, screenshot_path)

        steps_since_last = 0
        regenerate_target()
        trail.clear()

    draw(loss, reward, cx, cy, successes, steps_since_last)
    clock.tick(60)

pygame.quit()
