# Diffusion Limited Aggregation in Vivarium

From https://en.wikipedia.org/wiki/Diffusion-limited_aggregation --- Diffusion-limited aggregation (DLA) is the process whereby particles undergoing a random walk due to Brownian motion cluster together to form aggregates of such particles.

From http://paulbourke.net/fractals/dla/: There are a number of ways of simulating this process by computer, perhaps the most common is to start with a white image except for a single black pixel in the center. New points are introduced at the borders and randomly (approximation of Brownian motion) walk until they are close enough to stick to an existing black pixel. A typical example of this is shown below in figure 1. If a point, during its random walk, approaches an edge of the image there are two strategies. The point either bounces off the edge or the image is toroidally bound (a point moving off the left edge enters on the right, a point moving off the right edge enters on the left, similarly for top and bottom). In general new points can be seeded anywhere in the image area, not just around the border without any significant visual difference.

https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.47.1400
Witten, T.A. and Sander, L. M. and published by them in 1981, titled: "Diffusion limited aggregation, a kinetic critical phenomena" in Physical Review Letters. number 47.

# Useful for testing distributed job features, and for simulating retinal vascular structures

E.G.

```
time simulate run configurations/dla.yaml --pdb
```

or

```
time psimulate run --pdb --project proj_simscience configurations/dla.yaml configurations/branches_dla.yaml -o /mnt/share/homes/abie/vivarium_results/
```
