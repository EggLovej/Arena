"""Unfitted concentrated-attack rival; excluded from the model library."""
from ais._atlas_policy import policy

def choose_action(state):
 return policy(state,reserve=.1,buffer=0.,holding=0.,saving=False,horizon=18.,costscale=40.,enemy=4.,surge=1.,ffa=False)
