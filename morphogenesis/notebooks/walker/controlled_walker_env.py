import pygame
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim

# --- Parameters ---
WIDTH, HEIGHT = 2000, 1200
dt = 0.9
globalDamping = 0.995
baseFrictionHigh = 0.9
baseFrictionLow = 0.05
springK = 0.25
contractionAlpha = 0.55
contractionSpeed = 0.05

THRESHOLD = 0.2
K_DEPTH = 2
BATCH_SIZE = 16
NOISE_STD = 0.05

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# --- Target ---
target_radius = 80
target_x = random.randint(200, WIDTH - 200)
target_y = random.randint(200, HEIGHT - 200)

# --- Controller Model ---
class ControllerModel(nn.Module):
    def __init__(self, in_dim=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

model = ControllerModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_buffer = []

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
        self.signal = 0.0

    def accumulated_channels(self, depth=K_DEPTH):
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
                if e.a == idx:
                    dfs(e.b, d + 1)
                elif e.b == idx:
                    dfs(e.a, d + 1)

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
def update():
    # Update contractions
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

    # Springs
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

    # Friction handling
    for e in edges:
        if e.phase > 0:
            base = nodes[e.a]
            lifted = nodes[e.b]
            base.friction = baseFrictionHigh*(1 - 0.5*e.phase) + baseFrictionLow*(0.5*e.phase)
            lifted.friction = baseFrictionHigh*(1 - e.phase) + baseFrictionLow*e.phase

    # Motion update
    for n in nodes:
        n.vx *= n.friction
        n.vy *= n.friction
        n.vx *= globalDamping
        n.vy *= globalDamping
        n.x += n.vx*dt
        n.y += n.vy*dt

# --- Center of mass trail ---
trail = []

def update_trail():
    cx = sum(n.x for n in nodes)/len(nodes)
    cy = sum(n.y for n in nodes)/len(nodes)
    trail.append((cx, cy))

def center_of_mass_distance():
    cx = sum(n.x for n in nodes)/len(nodes)
    cy = sum(n.y for n in nodes)/len(nodes)
    dx = cx - target_x
    dy = cy - target_y
    return math.sqrt(dx*dx + dy*dy)

# --- Drawing ---
def draw(loss_val):
    screen.fill((10, 10, 20))

    # trail
    if len(trail) > 1:
        pygame.draw.lines(screen, (255,255,255), False, trail, 2)

    # target
    target_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.circle(target_surface, (255, 0, 0, 80), (target_x, target_y), target_radius)
    screen.blit(target_surface, (0, 0))

    # edges
    for e in edges:
        n1, n2 = nodes[e.a], nodes[e.b]
        color = (200, 50, 50) if e.phase > 0 else (150,150,150)
        pygame.draw.line(screen, color, (n1.x, n1.y), (n2.x, n2.y), 2)

    # nodes
    for n in nodes:
        f = (n.friction-baseFrictionLow)/(baseFrictionHigh-baseFrictionLow)
        f = max(0, min(1, f))
        r = 30*(1-f)+20*f
        g = 200*f+80*(1-f)
        b = 200*(1-f)+120*f
        pygame.draw.circle(screen, (int(r),int(g),int(b)), (int(n.x),int(n.y)), n.radius)

    # COM
    cx, cy = trail[-1]
    pygame.draw.circle(screen, (255,255,255), (int(cx),int(cy)), 3)

    # loss text
    font = pygame.font.SysFont(None, 30)
    loss_text = font.render(f"Loss: {loss_val:.4f}", True, (255,255,255))
    dist_text = font.render(f"COM Distance: {center_of_mass_distance():.2f}", True, (255,255,255))
    screen.blit(loss_text, (20, 20))
    screen.blit(dist_text, (20, 50))

    pygame.display.flip()

# --- Main loop ---
buildConnectedGraph()

running = True
loss_val = 0.0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Model inference ---
    raw_signals = []
    for n in nodes:
        acc = n.accumulated_channels()
        x = torch.tensor(acc, dtype=torch.float32)
        with torch.set_grad_enabled(True):
            signal = model(x)
        noise = torch.normal(mean=0.0, std=NOISE_STD, size=signal.shape)
        signal = signal + noise
        n.signal = signal.item()
        raw_signals.append(signal)

    # --- Edge contraction decisions ---
    for e in edges:
        s1 = nodes[e.a].signal
        s2 = nodes[e.b].signal
        if abs(s1 - s2) > THRESHOLD:
            if s1 < s2:
                contract_edge(e.a, e.b)
            else:
                contract_edge(e.b, e.a)

    # --- Surrogate loss (keep graph) ---
    raw_signals = torch.stack(raw_signals)
    loss_val = -raw_signals.mean()  # encourage stronger contractions

    loss_buffer.append(loss_val)
    if len(loss_buffer) >= BATCH_SIZE:
        batch_loss = torch.stack(loss_buffer).mean()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loss_buffer = []

    # --- Update & draw ---
    update()
    update_trail()
    draw(loss_val.item())
    clock.tick(60)

pygame.quit()
