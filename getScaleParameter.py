import gurobipy as gp
from gurobipy import GRB
import numpy as np
from enum import Enum
class Center(Enum):
    Start = 0
    Center = 1
    End = 2
    
class Goal(Enum):
    Upper = 0
    Lower = 1


def getScaleParameter(
    imageExtremeValue: np.ndarray, 
    center: Center,
    goal:Goal) -> tuple[float,float,float]:
    
    center_axis = len(imageExtremeValue) // 2
    print("center_axis:",center_axis)
    model = gp.Model("quadratic_area")

    a = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="a")
    b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="b")
    c = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="c")

    n_segments = 5
    area_vars = []
    for i in range(n_segments):
        area_var = model.addVar(lb=0, name=f"area_{i}")
        area_vars.append(area_var)

    model.setObjective(gp.quicksum(area_vars), GRB.MINIMIZE)

    for i in range(imageExtremeValue.shape[0]):
        y = imageExtremeValue[i]
        x = i
        if goal == Goal.Upper:
            model.addConstr(a * x ** 2 + b * x + c >= y, f"point_{x}_{y}")
        else:
            model.addConstr(a * x ** 2 + b * x + c <= y, f"point_{x}_{y}")
        
    if goal == Goal.Upper:
        model.addConstr(a >= 0, "a_positive")
    else:
        model.addConstr(a <= 0, "a_negative")
        
    #set a center axis
    model.addConstr( -b ==center_axis* (2*a), "set_center_axis")

    x_max = len(imageExtremeValue)
    segment_width = x_max / n_segments
    for i in range(n_segments):
        x_start = i * segment_width
        x_end = (i + 1) * segment_width
        y_start = a * x_start ** 2 + b * x_start + c
        y_end = a * x_end ** 2 + b * x_end + c
        if goal == Goal.Upper:
            model.addConstr(area_vars[i] >= (y_start + y_end) / 2 * segment_width, f"area_{i}")
        else:
            model.addConstr(area_vars[i] >= -(y_start + y_end) / 2 * segment_width, f"area_{i}")

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        a_val = a.x
        b_val = b.x
        c_val = c.x

        print("二次函数参数：")
        print(f"a: {a_val}")
        print(f"b: {b_val}")
        print(f"c: {c_val}")

        import matplotlib.pyplot as plt
        x = np.arange(0, len(imageExtremeValue), 0.01)
        y = a_val * x ** 2 + b_val * x + c_val
        plt.plot(x, y, label="Quadratic curve")
        plt.scatter(range(len(imageExtremeValue)), imageExtremeValue, color="red", label="Points")
        plt.legend()
        plt.show()
        return a_val, b_val, c_val
    

    else:
        print("模型未找到可行解，请检查输入数据和模型设置。")


if __name__ == '__main__':
    image = np.array([3,2,1,0,1,2,3,5,7,-9])
    getScaleParameter(image, Center.Center, Goal.Upper)
