import geatpy as ea
import numpy as np
import math

#
# part 1 : parameters used in the allocation problem
#
# Bombs
d_b1 = 100. # damage of bomb, b1 
d_b2 = 300. # damage of bomb, b2 

# Targets
dp_t1 = 10.  # damage point of target, t1
dp_t2 = 300. # damage point of target, t2


v_t1 = 10000. # value of target, t1
v_t2 = 100.  # value of target, t2

sigma_t1 = 50. # sigma of the target, t1
sigma_t2 = 50. # sigma of the target, t2

#
# part 2 : parameters of the mission
#
# assets : bombs
d_a1 = d_b1 # damage of asset, a1, bomb, b1 
d_a2 = d_b1 # damage of asset, a2, bomb, b1 
d_a3 = d_b2 # damage of asset, a3, bomb, b2 


def Gaussian(x, mu, sigma):
    G = 0

    G_part1 = ((2.*math.pi)**0.5) * sigma
    G_part1 = 1./G_part1

    G_part2 = -1.* ((x-mu)**2) / (2.*sigma**2)
    G_part2 = math.exp(G_part2)

    G = G_part1 * G_part2


    return G



def HalfGaussian(x, mu, sigma):
    G = 0

    # Situation 1 : x < mu
    if x<=mu:
        G_part1 = ((2.*math.pi)**0.5) * sigma
        G_part1 = 1./G_part1
    
        G_part2 = -1.* ((x-mu)**2) / (2.*sigma**2)
        G_part2 = math.exp(G_part2)
    
        G = G_part1 * G_part2

    # Situation 2 : x > mu
    else:
        G_part1 = ((2.*math.pi)**0.5) * sigma
        G_part1 = 1./G_part1

        G = G_part1


    return G



# 构建问题
r = 1  # 目标函数需要用到的额外数据

@ea.Problem.single
def evalVars(Vars):  # 定义目标函数（含约束）

    m_a1_t1 = Vars[0]
    m_a2_t1 = Vars[1]
    m_a3_t1 = Vars[2]

    m_a1_t2 = Vars[3]
    m_a2_t2 = Vars[4]
    m_a3_t2 = Vars[5]


    # D_t1
    D_t1 = 0
    D_t1 = D_t1 + m_a1_t1*d_a1 
    D_t1 = D_t1 + m_a2_t1*d_a2 
    D_t1 = D_t1 + m_a3_t1*d_a3 

    f_profit_part1 = v_t1*HalfGaussian(D_t1, dp_t1, sigma_t1)

    # D_t2
    D_t2 = 0
    D_t2 = D_t2 + m_a1_t2*d_a1 
    D_t2 = D_t2 + m_a2_t2*d_a2 
    D_t2 = D_t2 + m_a3_t2*d_a3 

    f_profit_part2 = v_t2*HalfGaussian(D_t2, dp_t2, sigma_t2)

    # f_profit 
    f_profit = f_profit_part1 + f_profit_part2

    # constraints
    CV = np.array([ m_a1_t1+m_a1_t2-1,
                    -m_a1_t1-m_a1_t2,
                    m_a2_t1+m_a2_t2-1,
                    -m_a2_t1-m_a2_t2,
                    m_a3_t1+m_a3_t2-1,
                    -m_a3_t1-m_a3_t2])  # 计算违反约束程度

    return f_profit, CV

if __name__=='__main__':


    problem = ea.Problem(name='soea quick start demo',
                            M=1,  # 目标维数
                            maxormins=[-1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                            Dim=6,  # 决策变量维数
                            varTypes=[1, 1, 1, 1, 1, 1],  # 决策变量的类型列表，0：实数；1：整数
                            lb=[0, 0, 0, 0, 0, 0],  # 决策变量下界
                            ub=[1.1, 1.1, 1.1, 1.1, 1.1, 1.1],  # 决策变量上界
                            evalVars=evalVars)
    # 构建算法
    algorithm = ea.soea_SEGA_templet(problem,
                                        ea.Population(Encoding='RI', NIND=20),
                                        MAXGEN=5000,  # 最大进化代数。
                                        logTras=100,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                        trappedValue=1e-12,  # 单目标优化陷入停滞的判断阈值。
                                        maxTrappedCount=10)  # 进化停滞计数器最大上限值。
    # 求解
    res = ea.optimize(algorithm, seed=1, verbose=True, drawing=0, outputMsg=True, drawLog=False, saveFlag=True, dirName='result')

    # Result Analysis
    Vars = res['Vars'][0]

    value = 0   
    value = value + Vars[0]*v_t1
    value = value + Vars[1]*v_t1
    value = value + Vars[2]*v_t1
    value = value + Vars[3]*v_t2
    value = value + Vars[4]*v_t2
    value = value + Vars[5]*v_t2

    print('value: ', value)
