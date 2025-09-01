import pygame
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Parameters ---
WIDTH, HEIGHT = 2000, 1200
dt = 0.9
globalDamping = 0.995
baseFrictionHigh = 0.9
baseFrictionLow = 0.05
springK = 0.25
contractionAlpha = 0.55
contractionSpeed = 0.05

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# --- Target ---
target_radius = 80
target_x = random.randint(200, WIDTH - 200)
target_y = random.randint(200, HEIGHT - 200)

# --- Controller Model ---
class NodeController(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_signal = nn.Linear(hidden_dim, 1)
        self.out_delta = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        signal = torch.sigmoid(self.out_signal(h))
        delta = torch.tanh(self.out_delta(h))
        return signal, delta

controller = NodeController()  # just instantiated, not applied yet

# --- Classes ---
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.radius = 6
        self.friction = baseFrictionLow

        # 5 information channels
        self.channels = [random.uniform(-1, 1) for _ in range(5)]

    def accumulated_channels(self, depth=2):
        """Return sum of sigmoids of channel values from neighbors up to given depth."""
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

def add_node():
    if not nodes: return
    base = random.choice(nodes)
    angle = random.random()*2*math.pi
    r = 60 + random.random()*40
    nx = base.x + math.cos(angle)*r
    ny = base.y + math.sin(angle)*r
    nodes.append(Node(nx, ny))
    new_index = len(nodes)-1
    edges.append(Edge(new_index, nodes.index(base)))

def add_edge():
    if len(nodes) < 2: return
    a, b = random.sample(range(len(nodes)), 2)
    if not any((e.a==a and e.b==b) or (e.a==b and e.b==a) for e in edges):
        edges.append(Edge(a, b))

def contract_edge(a, b):
    for e in edges:
        if e.a==a and e.b==b:
            e.contracting = True
            break

# --- Physics ---
def update():
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

# --- Trail ---
trail = []

def update_trail():
    cx = sum(n.x for n in nodes)/len(nodes)
    cy = sum(n.y for n in nodes)/len(nodes)
    trail.append((cx, cy))

# --- Drawing ---
def draw(loss=0.0):
    screen.fill((10, 10, 20))

    # trail
    if len(trail) > 1:
        pygame.draw.lines(screen, (255,255,255), False, trail, 2)

    # target circle
    target_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.circle(target_surface, (255, 0, 0, 80), (target_x, target_y), target_radius)
    screen.blit(target_surface, (0, 0))

    # edges
    for e in edges:
        n1, n2 = nodes[e.a], nodes[e.b]
        color = (200,50,50) if e.phase>0 else (150,150,150)
        pygame.draw.line(screen, color, (n1.x,n1.y), (n2.x,n2.y), 2)

    # nodes
    for i, n in enumerate(nodes):
        f = (n.friction-baseFrictionLow)/(baseFrictionHigh-baseFrictionLow)
        f = max(0, min(1, f))
        r = 30*(1-f)+20*f
        g = 200*f+80*(1-f)
        b = 200*(1-f)+120*f
        pygame.draw.circle(screen, (int(r),int(g),int(b)), (int(n.x),int(n.y)), n.radius)
        pygame.draw.rect(screen, (40,40,40), (n.x-10, n.y+10, 20, 4))
        pygame.draw.rect(screen, (255,255,255), (n.x-10, n.y+10, 20*f, 4))

    # center of mass dot
    cx, cy = trail[-1]
    pygame.draw.circle(screen, (255,255,255), (int(cx),int(cy)), 3)

    # loss text
    font = pygame.font.SysFont("Arial", 24)
    loss_text = font.render(f"Loss: {loss:.3f}", True, (255,255,255))
    screen.blit(loss_text, (20,20))

    # distance of mass center to target
    dx = cx - target_x
    dy = cy - target_y
    dist_to_target = max(0, math.sqrt(dx*dx + dy*dy) - target_radius)
    dist_text = font.render(f"Mass center dist: {dist_to_target:.1f}", True, (255,255,0))
    screen.blit(dist_text, (20, 50))

    pygame.display.flip()


# --- Main loop ---
buildConnectedGraph()
running = True
dummy_loss = 0.0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                if edges:
                    e = random.choice(edges)
                    contract_edge(e.a, e.b)
            elif event.key == pygame.K_n:
                add_node()
            elif event.key == pygame.K_e:
                add_edge()
    if edges:
        e = random.choice(edges)
        contract_edge(e.a, e.b)

    update()
    update_trail()
    draw(dummy_loss)
    clock.tick(60)

pygame.quit()
