import numpy as np
import gurobipy as gp
from gurobipy import GRB

def iterSequQBlock(vx, mW, vC):
    """
    Args:
        vx: Input vector.
        mW: Weight matrix.
        vC: Constant/Init vector.
        sL: Number of Decisions
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
    """
    nVars = mW.shape[1]
    #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQP)
    model = gp.Model("MIQCQP")
    # model.setParam('MIPGap', 1e-9)  # Sets the MIP gap to 0.01%
    # model.setParam("NumericFocus", 3)  # Increase numerical focus

    # Decision variables (vb) as binary, mapped to {-1, 1} in the objective
    vb = model.addVars(nVars, vtype=GRB.BINARY, name="vb")

    model.update()

    # Objective function
    objective = gp.QuadExpr()
    for i in range(mW.shape[0]):  # Rows of mW
        for j in range(nVars):  # Elements of vb and vx
            # Adjust the vb[j] from {0, 1} to {-1, 1}
            vbDec = 2*vb[j] - 1
            # Contribution of each element to the quadratic term
            sE = vx[j] - mW[i, j] * vbDec
            objective.add(sE * sE)

    # Set the objective
    model.setObjective(objective, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()

    # Output the solution
    if model.status == GRB.OPTIMAL:
        vb_out = np.array([2 * vb[j].X - 1 for j in range(nVars)])
        ve = mW @ vx - mW @ vb_out  # Ensure dimensions align
    else:
        print("No optimal solution found.")
        vb_out, ve = None, None
   
    return vb_out, ve