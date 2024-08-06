import numpy as np
import gurobipy as gp
import random
from gurobipy import GRB


def OptBlock(vx, mW, vE_hat, sL2Err):
    """
    Args:
        vx:         Input vector.
        mW:         Weight matrix.
        vE_hat:     Previous convolutional Error.
        vBStart:    Initial Decision 
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
    """
    snVars = mW.shape[1]
    snRows = mW.shape[0]
    #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQP)
    model = gp.Model("MIQCP")
    model.setParam("OutputFlag", 0)  # 0 to Suppress Gurobi output
    model.setParam("TimeLimit",10)
    model.setParam("VarBranch", 3)
    model.setParam("MIPFocus", 3)  # Shift focus to finding good feasible solutions quickly
    model.setParam("Heuristics", 0.3)  # Increase heuristic efforts
    model.setParam("Presolve", 0)  # More aggressive presolve
    
    # Decision variables (vb) as binary, mapped to {-1, 1} in the objective
    vb = model.addVars(snVars, vtype=GRB.BINARY, name="vb")
    #for j in range(snVars):
    #     vb[j].Start = 0
    ve = model.addVars(snRows, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="ve")     

    model.update()

    # Objective function
    obj = gp.QuadExpr()
    obj += sL2Err
 
    for i in range(snRows):  # Rows of mW
        se = 0
        for j in range(snVars):  # Elements of vb and vx
            # Adjust the vb[j] from {0, 1} to {-1, 1}
            vbDec = 2*vb[j] - 1
            sd = vx[j] - vbDec
            # Contribution of each element to the quadratic term
            se = se + mW[i, j] * sd
        
        se = se + vE_hat[i]
        model.addConstr(ve[i] == se)     
        obj += ve[i] * ve[i] 

    # Set the objective
    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()

    # Output the solution
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.", end=" ")
        vb_out = np.array([2 * vb[j].X - 1 for j in range(snVars)])
        ve_out = mW @ (vx - vb_out) + vE_hat
    else:
        print("No optimal solution found.", end=" ")
        vb_out = np.array([2 * vb[j].X - 1 for j in range(snVars)])
        ve_out = mW @ (vx - vb_out) + vE_hat 
    
    return vb_out, ve_out