# Dragon-Kings
A package for simulating dragon-king events over different network topologies.

# TODO
- [ ] Add `plotter` module to render output plots of the simulation.
- [ ] Fix `#TODO` tags across package.
- [ ] Patch bugs in status count procedure.
- [ ] Refactor `model` module to incorporate NetworkX.


# Current TODO
- [ ] Remove redundant final iteration of while-loop
- [ ] Patch spreading bug where status 2 nodes can still fail
- [x] Patch spreading bug where spread originates from wrong nodes (seemingly +-1 from actual failed node, not always)
- [ ] Add plotting of the first iteration