from walls import gen_wall
from tag import tag

res = gen_wall()
print(res)
#modes of movement of seeker and runner changeable within code in tag.py

if res:
    iteration=int(input("how many iterations would you like to run?:"))
    for _ in range(iteration):
        tag(res)
else:
    print("quitted/invalid")