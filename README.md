# Takt Optimization and BIM Viewer

This repository contains a preliminary takt planning optimization workflow and an interactive frontend for reviewing optimization results against a BIM model. The current version is an early prototype focused on Level 1 (L1) of the building. It is intended as a starting point for evaluation, discussion, and iteration, and feedback is very welcome.

- [Website Viewer](https://ashjs2003.github.io/Takt_Plan_Optimizer/).
- [Project Report of Stanford CS361](./AA222_Final_Project.pdf)
- The building BIM model was developed as a part of Stanford CEE 222. The video presentation of the building is [here](https://youtu.be/hFlzPlj1MCY)

## Project Structure

- `src/`: optimizer, candidate generation, scheduling, and logging code.
- `data/`: BIM-derived room, quantity, productivity, and equipment inputs.
- `outputs/`: generated optimizer outputs, plots, candidate logs, takt plans, and zoning images.
- `docs/`: frontend UI for viewing the Pareto front, model viewer, takt plan, and zoning.

## Frontend Features

- Final-generation Pareto front for duration versus crew cost.
- Candidate metrics, crew allocation, and material/equipment summary.
- Live takt plan that updates when a Pareto point is selected.
- 3D model viewer that updates zone coloring based on the selected candidate.
- Takt zoning view for the selected candidate.
- Crew idle time table for the selected candidate.

## Current Scope and Limitations

This is a preliminary version for L1 of the building only. The workflow assumes the current L1 room, quantity, and productivity data in `data/`.

Future improvements could include broader building levels, richer model-element mappings, more robust frontend data generation, and additional validation of optimizer outputs.
