import gym
from gym import spaces
import numpy as np
from random import randint
import math
import pygame
from PIL import Image
import sys

class Entity():
    def __init__(self,x,y,size,color="black",name=""):
        self.coords=[x,y,size,size]
        self.prev_pos=[x,y]
        self.color = color
        self.start_coords = [x,y]
        self.size = size
        self.vx,self.vy=0,0
        self.name=name
        self.sprite = pygame.Rect(*self.coords)
    def draw(self, window):
        draw_rect(self.coords,self.color,0, window)
    def move(self):
        self.coords[0]+=self.vx
        self.coords[1]+=self.vy
    def draw_rect(self, c, window, color="black", w=1, ):
        #print(c[0],c[1],c[2],c[3])    
        self.sprite = pygame.Rect(c[0],c[1],c[2],c[3])
        pygame.draw.rect(window,pygame.Color(color),self.sprite,w)

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    self.res = gen_wall()
    self.window=pygame.display.set_mode((300,300))
    self.x = 0
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(4)
    #self.action_space = spaces.Box( np.array([0,0,0,0,0]), np.array([+1,+1,+1,+1,+1]))
    # Example for using image as input:
    self.observation_space = spaces.Box(np.zeros((4,60, 60)), np.full((4,60, 60), 256), dtype=np.int8)
    '''self.seeker=Entity(95,150,10,"red","seeker")
    self.runner=Entity(190,150,10,"green","runner")'''
    while True:
        c=[randint(0,275),randint(0,275)]
        if not any([rect_collision(c+[25,25],n) for n in self.res]):
            break
    self.seeker=Entity(c[0],c[1],25,"red","seeker")
    while True: 
        c = [randint(0,285),randint(0,285)]
        if not any([rect_collision(c+[15,15],n) for n in self.res+[self.seeker.coords]]):
            break
    #c = [150,150]
    self.runner=Entity(c[0],c[1],15,"green","runner")

  def step(self, action1, action2):
    reward1 = 0
    reward2 = 0
    rewards = []
    done = False
    info = {}
    frames = []
    for x in range(4):
        if not done:
            self.seeker.prev_pos=self.seeker.coords[:2]
            self.runner.prev_pos=self.runner.coords[:2]

            #update velocity of each entity
            key_move(action2, self.seeker)
            #key_move(action1,self.runner)
            resistance(self.runner)
            resistance(self.seeker)

            #move entities
            self.seeker.move()
            #self.runner.move()

            #bound within screen, momentum reset if border hit
            self.seeker.coords=out_of_bounds(self.seeker)
            self.runner.coords=out_of_bounds(self.runner)

            wall_phys(self.runner, self.res)
            wall_phys(self.seeker, self.res)

            #check runner being caught
            #rect_collision(self.seeker.coords, self.runner.coords)
            if self.seeker.sprite.colliderect(self.runner.sprite):
                if not done:
                    reward2 += 100
                done = True
                
            #else:'''
            if not done:
                distance_away = distance_changed(self.seeker, self.runner)
                #reward1 += distance_away
                reward2 += distance_away
                
        self.render()
        frame = Image.open("frame"+str(x)+".png")
        frame = np.array(frame)
        frame = preprocess(frame)
        frames.append(frame)
    
    
    observation = np.array([frames])
    rewards = [reward1, reward2]
    return observation, rewards, done, info

  def reset(self):
    while True:
        c=[randint(0,275),randint(0,275)]
        if not any([rect_collision(c+[25,25],n) for n in self.res]):
            break
    self.seeker=Entity(c[0],c[1],25,"red","seeker")
    while True: 
        c = [randint(0,285),randint(0,285)]
        if not any([rect_collision(c+[15,15],n) for n in self.res+[self.seeker.coords]]):
            break
    #c = [150,150]
    self.runner=Entity(c[0],c[1],15,"green","runner")


    observation = np.zeros((4,60, 60))
    return observation  # reward, done, info can't be included

  def render(self, mode='human'):
    self.window.fill(pygame.Color("white"))

    for n in self.res:
        draw_rect(self.window, c=n, w=0)

    self.seeker.draw_rect(self.seeker.coords,self.seeker.color,0, self.window)
    self.runner.draw_rect(self.runner.coords,self.runner.color,0, self.window)

    pygame.display.flip()
    pygame.image.save(window, "frame"+str(x%4)+".png")
    self.x+=1

  def close (self):
    return

def draw_rect(c, color="black", w=1, window):
    pygame.draw.rect(window,pygame.Color(color),(c[0],c[1],c[2],c[3]),w)

def preprocess(frame):
    return np.mean(frame[::5, ::5], axis=2).astype(np.uint8)

def distance_changed(seeker, runner):
    cx = seeker.coords[0] - runner.coords[0]
    cy = seeker.coords[1] - runner.coords[1]
    px = seeker.prev_pos[0] - runner.prev_pos[0]
    py = seeker.prev_pos[1] - runner.prev_pos[1]
    cd = math.sqrt(pow(cx, 2)+pow(cy, 2))
    pd = math.sqrt(pow(px, 2)+pow(py, 2))
    return pd - cd


def limitingFunc(x): #limiting function to emulate true acceleration
    return pow(math.e, -(x/2.2))

def rect_collision(rect1,rect2): #rect1 = [x,y,x+dx,y+dy]
    if (((rect1[0] <= rect2[0]+rect2[2]) and (rect1[0] >= rect2[0])) or ((rect1[0]+rect1[2] <= rect2[0]+rect2[2]) and (rect1[0]+rect1[2] >= rect2[0]))):
        if (((rect1[1] <= rect2[1]+rect2[3]) and (rect1[1] >= rect2[1])) or ((rect1[1]+rect1[3] <= rect2[1]+rect2[3]) and (rect1[1]+rect1[3] >= rect2[1]))):
            return True

def out_of_bounds(i): #r = Rect
    r=i.coords
    x=False
    y=False
    if r[0]<=0: 
        r[0] = 0
        x=True
    elif r[0]+r[2]>=300: 
        r[0]=300-r[2]
        x=True
    if r[1]<=0:
        r[1]=1
        y=True
    elif r[1]+r[3]>=300: 
        r[1]=300-r[3]
        y=True
    if x: 
        i.vx=0
    if y: 
        i.vy=0
    return r

def wall_phys(entity, res):
    for n in res:
        if rect_collision(entity.coords,n):
            t=entity.coords
            if rect_collision([entity.prev_pos[0]]+t[1:],n): #if reverting x fails
                if rect_collision([t[0]]+[entity.prev_pos[1]]+t[2:],n): #if reverting y fails
                    entity.coords=entity.prev_pos+[entity.size,entity.size] #revert both
                    entity.vx,entity.vy=0,0
                else: #revert only y
                    entity.coords=[t[0],entity.prev_pos[1],entity.size,entity.size]
                    entity.vy=0
            else: #revert only x
                entity.coords=[entity.prev_pos[0],t[1],entity.size,entity.size]
                entity.vx=0
            break

def key_move(key,i): #key mode, i=instance
    key = int(key)
    if key == 1:
        i.vy -= 1.5*limitingFunc(0.5*(abs(i.vy)+0.5))
    elif key == 2:
        i.vy += 1.5*limitingFunc(0.5*(abs(i.vy)+0.5))
    elif key == 3:
        i.vx -= 1.5*limitingFunc(0.5*(abs(i.vx)+0.5))
    elif key == 0:
        i.vx += 1.5*limitingFunc(0.5*(abs(i.vx)+0.5))
    else:
        print('invalid')

def resistance(i): #basic resistance to make the acceleration feel better
    if i.vy < -(0.1):
        i.vy += 0.1
    elif i.vy > 0.1:
        i.vy -= 0.1
    else:
        i.vy = 0
    
    if i.vx < -(0.1):
        i.vx += 0.1
    elif i.vx > 0.1:
        i.vx -= 0.1
    else:
        i.vx = 0

def basic_move(a,b): #basic mode, a=moved instance, b=based instance
    [ax,ay]=a.coords[:2]
    [bx,by]=b.coords[:2]
    if ax<bx:
        a.vx+=1.1*limitingFunc(0.5*(abs(a.vx)+0.5))
    else:
        a.vx-=1.1*limitingFunc(0.5*(abs(a.vx)+0.5))
    #x
    if ay<by:
        a.vy+=1.1*limitingFunc(0.5*(abs(a.vy)+0.5))
    else:
        a.vy-=1.1*limitingFunc(0.5*(abs(a.vy)+0.5))

def gen_wall():
    import pygame
    import math

    window_x,window_y=300,300
    pygame.init()
    window=pygame.display.set_mode((window_x,window_y))

    wall_size=20 #divide 300x300 screen into 20x20 pixel blocks
    wall_rects=[]

    run=True; returned=False
    while run:
        pygame.time.delay(1)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run=False
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE:
                    run=False
                if event.key==pygame.K_RETURN:
                    returned=True
                    run=False
            if event.type==pygame.MOUSEBUTTONUP:
                (mx,my)=pygame.mouse.get_pos()
                x,y=math.floor(mx/wall_size),math.floor(my/wall_size)
                c=[x*wall_size,y*wall_size,wall_size,wall_size]
                if c in wall_rects:
                    wall_rects.remove(c)
                else:
                    wall_rects.append(c)

        window.fill(pygame.Color("white"))

        for n in wall_rects:
            draw_rect(c=n, w=0, window)

        pygame.display.flip()
    if returned:
        return wall_rects
    else:
        return []

pygame.init()
