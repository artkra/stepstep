import pygame
import random
import math

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

# --- Classes ---
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.radius = 6
        self.friction = baseFrictionLow  # start loose

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

    # Friction handling: base grips, lifted slides
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
trail = []  # now unlimited

def update_trail():
    cx = sum(n.x for n in nodes)/len(nodes)
    cy = sum(n.y for n in nodes)/len(nodes)
    trail.append((cx, cy))

# --- Drawing ---
def draw():
    screen.fill((10, 10, 20))

    # trail (all in white, persistent)
    if len(trail) > 1:
        pygame.draw.lines(screen, (255,255,255), False, trail, 2)

    # edges
    for e in edges:
        n1, n2 = nodes[e.a], nodes[e.b]
        color = (200, 50, 50) if e.phase>0 else (150,150,150)
        pygame.draw.line(screen, color, (n1.x, n1.y), (n2.x, n2.y), 2)

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

    pygame.display.flip()

# --- Main loop ---
buildConnectedGraph()

running = True
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
    draw()
    clock.tick(60)

pygame.quit()
