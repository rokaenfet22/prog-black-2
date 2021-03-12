def tag(wall_rects):

    import pygame
    from random import randint

    window_x,window_y=300,300
    pygame.init()
    window=pygame.display.set_mode((window_x,window_y))

    #modes of movement, key, random, basic
    seeker_move_mode="key"
    runner_move_mode="basic"
    #size of both
    seeker_size=10
    runner_size=10
    #momentum val
    acceleration=0.5

    class Entity():
        def __init__(self,x,y,size,color="black",name=""):
            self.coords=[x,y,size,size]
            self.prev_pos=[x,y]
            self.color=color
            self.vx,self.vy=0,0
            self.name=name
            self.size=size
        def draw(self):
            draw_rect(self.coords,self.color,0)
        def move(self):
            self.coords[0]+=self.vx
            self.coords[1]+=self.vy

    def draw_rect(c,color="black",w=1):
        pygame.draw.rect(window,pygame.Color(color),(c[0],c[1],c[2],c[3]),w)

    def rect_collision(rect1,rect2): #rect1 = [x,y,x+dx,y+dy]
        if ((rect1[0] < rect2[0]+rect2[2]) and (rect1[0] > rect2[0])) or ((rect1[0]+rect1[2] < rect2[0]+rect2[2]) and (rect1[0]+rect1[2] > rect2[0])):
            if ((rect1[1] < rect2[1]+rect2[3]) and (rect1[1] > rect2[1])) or ((rect1[1]+rect1[3] < rect2[1]+rect2[3]) and (rect1[1]+rect1[3] > rect2[1])):
                return True

    def out_of_bounds(i): #r = Rect
        r=i.coords
        x=False
        y=False
        if r[0]<=0: r[0] = 0; x=True
        elif r[0]+r[2]>=window_x: r[0]=window_x-r[2]; x=True
        if r[1]<=0: r[1]=1; y=True
        elif r[1]+r[3]>=window_y: r[1]=window_y-r[3]; y=True
        if x: i.vx=0
        if y: i.vy=0
        return r

    def key_move(key,i): #key mode, i=instance
        if key[pygame.K_w]: i.vy-=acceleration
        elif key[pygame.K_s]: i.vy+=acceleration
        if key[pygame.K_a]: i.vx-=acceleration
        elif key[pygame.K_d]: i.vx+=acceleration

    def random_move(key,i): #random mode, 0=move in negative, 1=move in positive, 2= don't move
        r=randint(0,2)
        if r==2: i.vy-=acceleration
        elif r: i.vy+=acceleration
        r=randint(0,2)
        if r==2: i.vx-=acceleration
        elif r: i.vx+=acceleration

    def basic_move(key,a,b): #basic mode, a=moved instance, b=based instance
        [ax,ay]=a.coords[:2]
        [bx,by]=b.coords[:2]
        if a.name=="seeker":
            #y
            if ax<bx:a.vx+=acceleration
            elif ax>bx:a.vx-=acceleration
            #x
            if ay<by:a.vy+=acceleration
            elif ay>by:a.vy-=acceleration
        elif a.name=="runner":
            #y
            if ax<bx:a.vx-=acceleration
            elif ax>bx:a.vx+=acceleration
            #x
            if ay<by:a.vy-=acceleration
            elif ay>by:a.vy+=acceleration

    def update(key,s,r): #s=seeker, r=runner
        if seeker_move_mode=="key":
            key_move(key,s)
        elif seeker_move_mode=="random":
            random_move(key,s)
        elif seeker_move_mode=="basic":
            basic_move(key,s,r)

        if runner_move_mode=="key":
            key_move(key,r)
        elif runner_move_mode=="random":
            random_move(key,r)
        elif runner_move_mode=="basic":
            basic_move(key,r,s)

    #create instances of seeker and runner, generate in vacant space
    while True:
        c=[randint(0,window_x-seeker_size),randint(0,window_y-seeker_size)]
        if not any([rect_collision(c+[seeker_size,seeker_size],n) for n in wall_rects]):
            break
    seeker=Entity(c[0],c[1],seeker_size,"red","seeker")
    while True:
        c = [randint(0,window_x-runner_size),randint(0,window_y-runner_size)]
        if not any([rect_collision(c+[runner_size,runner_size],n) for n in wall_rects+[seeker.coords]]):
            break
    runner=Entity(c[0],c[1],runner_size,"green","runner")

    run=True
    caught=False
    clock=pygame.time.Clock()
    fps=30
    while run:
        dt=clock.tick(fps)
        #dt=how many ms since last iteration/frame i.e. 30fps atm
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run=False
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE:
                    run=False

        #keep track of where it was previously
        seeker_temp_prev_pos=seeker.coords[:2]
        runner_temp_prev_pos=runner.coords[:2]

        #update velocity of each entity
        update(pygame.key.get_pressed(),seeker,runner)

        #move entities
        seeker.move()
        runner.move()

        #bound within screen, momentum reset if border hit
        seeker.coords=out_of_bounds(seeker)
        runner.coords=out_of_bounds(runner)

        #wall collision, momentum reset if wall hit
        for i in (seeker,runner):
            m=False
            for n in wall_rects:
                if rect_collision(i.coords,n):
                    m=True
                    t=seeker.coords
                    if rect_collision([i.prev_pos[0]]+t[1:],n): #if reverting x fails
                        if rect_collision([t[0]]+[i.prev_pos[1]]+t[2:],n): #if reverting y fails
                            i.coords=i.prev_pos+[i.size,i.size] #revert both
                            i.vx,i.vy=0,0
                        else: #revert only y
                            i.coords=[t[0],i.prev_pos[1],i.size,i.size]
                            i.vy=0
                    else: #revert only x
                        i.coords=[i.prev_pos[0],t[1],i.size,i.size]
                        i.vx=0
                    break
            if not m:
                if i.name=="seeker":
                    i.prev_pos=seeker_temp_prev_pos
                else:
                    i.prev_pos=runner_temp_prev_pos
        """
        for i in (seeker,runner):
            for n in wall_rects:
                if rect_collision(i.coords,n):
                    
                    w=[n[0],n[1],n[0]+n[2],n[1]+n[3]]
                    
                    if i.vx<0: #moving left
                        i.coords[0]=w[2]
                        i.vx=0
                    elif i.vx>0: #moving right
                        i.coords[0]=w[0]-i.size
                        i.vx=0
                    if i.vy<0: #moving up
                        i.coords[1]=w[3]
                        i.vy=0
                    elif i.vy>0: #moving down
                        i.coords[1]=w[1]-i.size
                        i.vy=0
                    break"""

        #check runner being caught
        if rect_collision(seeker.coords,runner.coords):
            caught=True
            run=False #quit game when caught for now

        window.fill(pygame.Color("white"))

        seeker.draw()
        runner.draw()

        for n in wall_rects:
            draw_rect(n)

        pygame.display.flip()
    pygame.quit()

    if caught: print("GOTCHA BITCH")
    else: print("uncaught sad face :/")