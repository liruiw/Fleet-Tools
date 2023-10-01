import argparse
import logging
import numpy as np
import os
import sys
import webbrowser

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    FindResourceOrThrow,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RigidTransform,
    Rgba,
    Role,
    Simulator,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SDF and its collision geometry in meshcat.")
    parser.add_argument("sdf", type=str, help="Path to SDF.")
    parser.add_argument("--noopen", action="store_true", help="Open webbrowser automatically.")

    args = parser.parse_args()

    sdf = os.path.abspath(args.sdf)
    if not os.path.exists(sdf):
        logging.error("No SDF found at %s" % sdf)
        sys.exit(-1)

    # Start the visualizer.
    meshcat = Meshcat()
    web_url = meshcat.web_url()
    print("Meshcat available at %s" % web_url)
    if not args.noopen:
        webbrowser.open_new(web_url)

    # Build a diagram to do visualization.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.01)
    # Add a single object into it.
    parser = Parser(plant)
    model_id = parser.AddModelFromFile(sdf)
    body_ids = plant.GetBodyIndices(model_id)
    for body_id in body_ids:
        body = plant.get_body(body_id)
        plant.WeldFrames(plant.world_frame(), body.body_frame())
    plant.Finalize()

    # Add two visualizers with the same meshcat instance:
    #  one for visual, and one for collision.
    visual_params = MeshcatVisualizerParams(
        role=Role.kIllustration, prefix="visual", default_color=Rgba(0.1, 0.6, 1.0, 0.3)
    )
    # visual_viz = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat, visual_params)
    visual_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, visual_params)
    collision_params = MeshcatVisualizerParams(
        role=Role.kProximity, prefix="collision", default_color=Rgba(0.8, 0.5, 0.2, 0.3)
    )
    # collision_viz = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat, collision_params)
    collision_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, collision_params)

    meshcat.ResetRenderMode()

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    simulator.set_target_realtime_rate(1.0)

    meshcat.AddButton("Stop Simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
    meshcat.DeleteButton("Stop Simulation")
