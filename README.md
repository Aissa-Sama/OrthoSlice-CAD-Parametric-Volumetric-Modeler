# OrthoSlice CAD: Parametric Volumetric Modeler 

A hardware-accelerated, lightweight Orthographic CAD engine built in Python designed to construct cross-sectional 3D Topological Meshes over reference image slices (MRI/DICOM approach).

## Technical Architecture
This project is built from scratch mimicking the foundation of industrial 3D software (like simple Blender or 3D Slicer architectures):
- **Core Engine:** Written entirely in Python.
- **Hardware Acceleration:** Custom `ModernGL` shaders natively injected into the GPU for pure VRAM computations (bypassing CPU rendering bottlenecks).
- **Window Management:** Handled by `PyGame` (Input Events, DeltaTime buffers).
- **Graphical Interface (GUI):** Advanced `Imgui_bundle` integration delivering a strict, CAD-like native toolbar experience with state memory.

## Features
- **Cross-Sectional Modeling:** Orthographic alignment (Top/Front/Side) to simulate Transversal, Coronal, and Sagittal viewpoints.
- **Parametric 2D Overlay:** ImGui capabilities to Load, Scale, Translate (X/Y) and apply direct OpenGL shader opacity properties to imported reference images dynamically mapped to geometric space. 
- **Topological Graph Memory System:** 
  - Sub-millimeter accurate **Shared Vertex/Nodes dictionary** (Prevents duplicate memory instantiation).
  - Explicit **Edge/Line Generation** mapped strictly to valid ID nodes.
  - Runtime **Triangle-Fan Solid Polygon Creation** rendering colored Faces from connected cyclic nodes.
- **CAD Controls:** Axis-locked local panning (W/A/S/D mapping relative to selected plane) and geometric center relocation algorithms. 
- **Macro-State History Undo:** An advanced Stack system that gracefully pops explicit wireframes or entirely obliterates generated polygons including traversing dependencies.

## Stack Requirements
- Python 3.1x
- `moderngl` (Raw Shader/VBO/VAO handling)
- `pygame` (Display Surface and Event Dispatching)
- `imgui-bundle` (C++ PyBind GUI)
- `glm` (Vector and Matrix Math Computations)
- `numpy` (Memory structure optimization for OpenGL Buffers)
