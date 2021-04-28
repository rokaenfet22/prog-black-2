import pygame
import numpy as np

class Runner(object):
    def __init__(self,x,y,color,size):
        self.x,self.y,self.color=x,y,color
        self.vx=0
        self.vy=0
        self.size=size
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        self.is_catcher=False
    def get_pos(self):
        return self.rect.x,self.rect.y
    def get_v(self):
        return self.vx,self.vy
    def get_size(self):
        return self.size
    def set_pos(self,x,y):
        self.rect.x=x
        self.rect.y=y
        self.x=x
        self.y=y
    def get_rect(self):
        return self.rect
    def accelerate(self,ax,ay):
        self.vx+=ax
        self.vy+=ay
    def reset_v(self):
        self.vx=0
        self.vy=0
    # def move(self,screen_size,walls):
    #     self.vx,self.vy=np.sign(self.vx),np.sign(self.vy)
    #
    #     has_collided1=self.move_single_axis(self.vx, 0,screen_size,walls)
    #
    #     has_collided=self.move_single_axis(0, self.vy,screen_size,walls)
    #
    #     return has_collided1|has_collided

    def move(self,screen_size,walls,dx,dy):
        has_collided1=self.move_single_axis(dx, 0,screen_size,walls)

        has_collided=self.move_single_axis(0, dy,screen_size,walls)

        return has_collided1|has_collided
    def move_single_axis(self, dx, dy,screen_size,walls):
        has_collied=False
        # Move the rect
        self.rect.x += dx
        self.rect.y += dy
        #collision with border
        if self.rect.x<0:
            self.rect.x=0
            self.vx=0
            has_collied=True
        if self.rect.x+self.size>screen_size[0]:
            self.rect.x=screen_size[0]-self.size
            self.vx=0
            has_collied=True
        if self.rect.y < 0:
            self.rect.y = 0
            self.vy=0
            has_collied = True
        if self.rect.y+self.size > screen_size[1]:
            self.vy=0
            self.rect.y = screen_size[1]-self.size
            has_collied=True

        #If you collide with a wall, move out based on velocity
        for wall in walls:
            wall_rect = pygame.Rect(wall[0], wall[1], wall[2], wall[3])
            if self.rect.colliderect(wall_rect):
                has_collied = True
                if dx > 0:
                    self.rect.right = wall_rect.left
                    self.vx=0
                if dx < 0:
                    self.rect.left = wall_rect.right
                    self.vx=0
                if dy > 0:
                    self.rect.bottom = wall_rect.top
                    self.vy=0
                if dy < 0:
                    self.rect.top = wall_rect.bottom
                    self.vy=0
        self.x=self.rect.x
        self.y=self.rect.y
        return has_collied
class Catcher(Runner):
    def __init__(self,x,y,color,size):
        super().__init__(x,y,color,size)
        self.is_catcher=True