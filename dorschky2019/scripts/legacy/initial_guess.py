from biosym.ocp import collocation
ocp = collocation.Collocation("dorschky_demo/configs/standing2d.yaml")
ocp.solve()

ocp_walking = collocation.Collocation("dorschky_demo/configs/walking2d_initial_guess.yaml")
ocp_walking.solve(visualize=True)