import pygame,os
from Player import Player,IT
from Game import Game


#Initialise pygame
os.environ["SDL_VIDEO_CENTERED"] = "1"
pygame.init()
# Set up the display
pygame.display.set_caption("Tag")
SCREEN_SIZE=(50,50)
screen = pygame.display.set_mode(SCREEN_SIZE)
clock = pygame.time.Clock()
#
#walls=[[x,y,width,heigh],[x,y,width,height]]
player_size=3
player_a=0.2
#setup of walls as ratios of screen size.4/75 is the wall thickness
wall_setup_1=[[1/6,1/6,1/6,4/75],[1/6,1/6,4/75,1/6],[0,0.5,1/3,4/75],[23/30,11/30,1/5,4/75],[23/30,1/10,4/75,4/15],[1/3,23/30,3/10,4/75],[19/30,3/5,4/75,1/6]]
walls1=[[w[0]*SCREEN_SIZE[0],w[1]*SCREEN_SIZE[0],w[2]*SCREEN_SIZE[0],w[3]*SCREEN_SIZE[0]] for w in wall_setup_1]
it_player_pos=(16.7,16.7)
player1_pos=(25,25)
print(walls1)
it_player=IT(it_player_pos[0],it_player_pos[1],[255,0,0],player_size)
player1=Player(player1_pos[0],player1_pos[1],[0,0,255],player_size)
game=Game(it_player,player1,screen,SCREEN_SIZE,clock,walls1,player_a)
game.run()
