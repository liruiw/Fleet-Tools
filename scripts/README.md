## File Processing
1. To process the tool folder from .obj file and then label keypoints. Run ```python scripts/generate_tool_sdf_from_mesh.py  assets/tools/hammer/Hammer/Hammer.obj --density 20 --size 0.15 --preview --select_keypoint```
2. To process the object folder from .obj file. Run ```python scripts/generate_object_sdf_from_mesh.py  assets/tool_obj/spatula/Plate --density 200 --size 0.1  --select_keypoint --rename_idx 5```
3. To select keypoints. Run ```python scripts/select_keypoint.py```
4. To visualize keypoints. Run ```python scripts/vis_keypoint.py --mesh_file assets/tools/wrench/model7/model7_corr.obj ```
