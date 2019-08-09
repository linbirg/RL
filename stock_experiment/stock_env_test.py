
from stock_env import StockEnv


env = StockEnv()

if __name__ == '__main__':
    env.render()
    # print(env.step(1))
    s,r,done = env.step(1)
    print(s)
    print(s.shape)
    print(r)

    print("=====================")
    s,r,done = env.step(0)
    print(s)
    print(r)

    print("====================")
    s,r,done = env.step(2)
    print(s)
    print(r)


    
