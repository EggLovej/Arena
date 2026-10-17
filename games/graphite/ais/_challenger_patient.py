"""Unfitted cautious rival; excluded from the model library."""
from ais._atlas_policy import policy

def choose_action(state):
 return policy(state,reserve=2.5,buffer=3.,holding=.8,saving=True,horizon=35.,costscale=15.,enemy=2.,surge=.5)
