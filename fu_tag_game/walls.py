def gen_wall():
    import pygame
    import math

    window_x,window_y=300,300
    pygame.init()
    window=pygame.display.set_mode((window_x,window_y))

    wall_size=20 #divide 300x300 screen into 20x20 pixel blocks
    wall_rects=[]

    def draw_rect(c,color="black",w=1):
        pygame.draw.rect(window,pygame.Color(color),(c[0],c[1],c[2],c[3]),w)

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
            draw_rect(n)

        pygame.display.flip()
    pygame.quit()

    if returned:
        return wall_rects
    else:
        return False