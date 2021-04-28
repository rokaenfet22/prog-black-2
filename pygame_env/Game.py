import pygame

class Game:
    def __init__(self,it_player,player,screen,screen_size,clock,wall_list,acceleration=0.5):
        self.it_player=it_player
        self.player=player
        self.wall_list=wall_list
        self.screen=screen
        self.clock=clock
        self.a=acceleration
        self.screen_size=screen_size

    def run(self):
        running = True
        while running:

            self.clock.tick(30)

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False

            # Move the player if an arrow key is pressed
            key = pygame.key.get_pressed()
            if key[pygame.K_LEFT]:
                self.it_player.accelerate(-self.a,0)
            if key[pygame.K_RIGHT]:
                self.it_player.accelerate(self.a, 0)
            if key[pygame.K_UP]:
                self.it_player.accelerate(0, -self.a)
            if key[pygame.K_DOWN]:
                self.it_player.accelerate(0, self.a)
            self.it_player.move(self.screen_size,self.wall_list)
            if self.it_player.rect.colliderect(self.player):
                raise SystemExit("IT tagged PLAYER")

            #Draw the scene
            self.screen.fill(pygame.Colour("white"))
            #fill screen excess black
            if self.screen.get_size()!=self.screen_size:
                r=pygame.Rect(self.screen_size[0],0,self.screen.get_size()[0]-self.screen_size[0],self.screen_size[1])
                pygame.draw.rect(self.screen,(0,0,0),r)
            for wall in self.wall_list:
                wall_rect=pygame.Rect(wall[0],wall[1],wall[2],wall[3])
                pygame.draw.rect(self.screen, (0, 0, 0), wall_rect)
            pygame.draw.rect(self.screen, self.it_player.color, self.it_player.rect)
            pygame.draw.rect(self.screen, self.player.color, self.player.rect)
            pygame.display.update()

