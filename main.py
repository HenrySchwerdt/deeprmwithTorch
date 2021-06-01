import enivronment
import parameters

pa = parameters.Parameters()
print(pa)
env = enivronment.Env(pa)
env.plot_state()