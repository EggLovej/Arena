"""Unfitted aggressive-economy rival; excluded from the opponent model library."""
from ais._atlas_policy import policy

def choose_action(state):
 return policy(state,reserve=.3,buffer=1.,holding=.15,saving=True,horizon=55.,costscale=3.,enemy=1.3,surge=.35)
