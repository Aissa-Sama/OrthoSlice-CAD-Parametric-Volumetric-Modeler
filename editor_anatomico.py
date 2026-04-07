import pygame
import moderngl
import numpy as np
import glm
import math
import struct
import tkinter as tk
from tkinter import filedialog

try:
    from imgui_bundle import imgui
    from imgui_bundle.python_backends.pygame_backend import PygameRenderer
    HAS_IMGUI = True
    BUNDLE = True
except ImportError:
    try:
        import imgui
        from imgui.integrations.pygame import PygameRenderer
        HAS_IMGUI = True
        BUNDLE = False
    except ImportError:
        HAS_IMGUI = False
        BUNDLE = False

# ════════════════════════════════════════════════════════════
# UNPROJECT (RAYCASTING)
# ════════════════════════════════════════════════════════════
def unproject(mouse_x: int, mouse_y: int, w: int, h: int, camara, plano: str, y_fijo: float, z_fijo: float, x_fijo: float):
    proj = camara.matriz_proyeccion(w, h)
    view = camara.matriz_vista()
    inv_mvp = glm.inverse(proj * view)

    x_ndc = (2.0 * mouse_x) / w - 1.0
    y_ndc = 1.0 - (2.0 * mouse_y) / h

    cerca  = inv_mvp * glm.vec4(x_ndc, y_ndc, -1.0, 1.0)
    lejos  = inv_mvp * glm.vec4(x_ndc, y_ndc,  1.0, 1.0)
    ray_o  = glm.vec3(cerca) / cerca.w
    ray_e  = glm.vec3(lejos) / lejos.w
    ray_d  = glm.normalize(ray_e - ray_o)

    if plano == "coronal":
        if abs(ray_d.z) < 1e-6: return None
        t = (z_fijo - ray_o.z) / ray_d.z
    elif plano == "sagital":
        if abs(ray_d.x) < 1e-6: return None
        t = (x_fijo - ray_o.x) / ray_d.x
    elif plano == "transversal":
        if abs(ray_d.y) < 1e-6: return None
        t = (y_fijo - ray_o.y) / ray_d.y
    else: return None
    
    if t < 0: return None
    hit = ray_o + ray_d * t
    
    # 1. Delimitar al área estricta (X:-30 a 30, Y:0 a 100, Z:-30 a 30)
    eps = 0.05
    if not (-30.0 - eps <= hit.x <= 30.0 + eps and 
              0.0 - eps <= hit.y <= 100.0 + eps and 
            -30.0 - eps <= hit.z <= 30.0 + eps):
        return None
        
    return hit.x, hit.y, hit.z

# ════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS / PLANOS INTERSECTADOS
# ════════════════════════════════════════════════════════════

class Plano:
    def __init__(self, tipo: str, coord: float):
        self.tipo = tipo  # 'transversal' (Y), 'coronal' (Z), 'sagital' (X)
        self.coord = coord
        self.activo = False

class Nodo:
    def __init__(self, id_nodo: int, x: float, y: float, z: float):
        self.id = id_nodo
        self.coord = (x, y, z)

class Arista:
    def __init__(self, id_a: int, n1: int, n2: int):
        self.id = id_a
        self.n1 = n1
        self.n2 = n2

class Cara:
    def __init__(self, id_c: int, nodos_ids: list, color: tuple):
        self.id = id_c
        self.nodos_ids = nodos_ids
        self.color = color

class VolumenAnatomico:
    def __init__(self):
        self.planos = []
        
        self.nodos = {}
        self.aristas = {}
        self.caras = {}
        self.cnt_id = 1
        
        self.historial = []
        
        self.modo_edicion = "linea" # "linea", "cara"
        self.color_actual = [0.6, 0.8, 0.2] # RGB float
        self.nodos_temp_cara = []
        self.nodo_start_linea = None
        self.cursor_3d = None
        
        self.vista_actual = "default"
        self.y_nivel = 50.0  
        self.x_nivel = 0.0
        self.z_nivel = 0.0
        self.esconder_inactivos = False
        self.densidad = 1
        self.snap_grid = True
        self.tamano_grid = 1.0
        
        self.ref_tex = None
        self.ref_aspect = 1.0
        self.ref_coord = 0.0
        self.ref_vista = "transversal"
        self.ref_scale = 1.0
        self.ref_offset_u = 0.0
        self.ref_offset_v = 0.0
        self.ref_opacity = 0.5
        
        self.generar_planos()

    def generar_planos(self):
        self.planos.clear()
        # Transversales (Y)
        for y in np.linspace(0, 100, 20 * self.densidad + 1):
            self.planos.append(Plano("transversal", y))
        # Coronales (Z)
        for z in np.linspace(-30, 30, 12 * self.densidad + 1):
            self.planos.append(Plano("coronal", z))
        # Sagitales (X)
        for x in np.linspace(-30, 30, 12 * self.densidad + 1):
            self.planos.append(Plano("sagital", x))

    def set_vista(self, vista: str):
        self.vista_actual = vista

    def add_nodo(self, coord):
        # Buscar duplicados por cercanía submilimétrica
        for n in self.nodos.values():
            if math.dist(n.coord, coord) < 0.1:
                return n.id, False
        nid = self.cnt_id
        self.nodos[nid] = Nodo(nid, *coord)
        self.cnt_id += 1
        self.historial.append(('nodo', nid))
        return nid, True
        
    def add_arista(self, n1, n2):
        if n1 == n2: return None
        for a in self.aristas.values():
            if (a.n1, a.n2) == (n1, n2) or (a.n1, a.n2) == (n2, n1):
                return a.id
        aid = self.cnt_id
        self.aristas[aid] = Arista(aid, n1, n2)
        self.cnt_id += 1
        self.historial.append(('arista', aid))
        return aid
        
    def add_cara(self, nodos_ids):
        new_aris = []
        for i in range(len(nodos_ids)):
            n1, n2 = nodos_ids[i], nodos_ids[(i+1)%len(nodos_ids)]
            exists = False
            for a in self.aristas.values():
                 if (a.n1, a.n2) == (n1, n2) or (a.n1, a.n2) == (n2, n1):
                     exists = True; break
            if not exists:
                 aid = self.cnt_id
                 self.aristas[aid] = Arista(aid, n1, n2)
                 self.cnt_id += 1
                 new_aris.append(aid)
        
        cid = self.cnt_id
        self.caras[cid] = Cara(cid, nodos_ids, tuple(self.color_actual))
        self.cnt_id += 1
        self.historial.append(('cara_macro', cid, new_aris))
        return cid
        
    def undo(self):
        if not self.historial: return
        act = self.historial.pop()
        tipo = act[0]
        if tipo == 'arista': self.aristas.pop(act[1], None)
        elif tipo == 'cara': self.caras.pop(act[1], None)
        elif tipo == 'nodo': self.nodos.pop(act[1], None)
        elif tipo == 'cara_macro':
            self.caras.pop(act[1], None)
            for aid in act[2]: self.aristas.pop(aid, None)

# ════════════════════════════════════════════════════════════
# RENDERIZADOR
# ════════════════════════════════════════════════════════════

VERTEX_SHADER = """
#version 330 core
in vec3 in_position;
in vec4 in_color;

uniform mat4 u_mvp;

out vec4 v_color;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_color = in_color;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in  vec4 v_color;
out vec4 fragColor;

void main() {
    fragColor = v_color;
}
"""

class Renderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        self.prog_tex = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 u_mvp;
                in vec3 in_position;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = u_mvp * vec4(in_position, 1.0);
                    v_uv = in_uv;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D Tex;
                uniform float u_opacity;
                in vec2 v_uv;
                out vec4 fragColor;
                void main() {
                    vec4 color = texture(Tex, v_uv);
                    color.a *= u_opacity;
                    fragColor = color;
                }
            '''
        )
        self._vbo: moderngl.Buffer | None = None
        self._vao: moderngl.VertexArray | None = None
        self._n_verts = 0
        
        self._vbo_lineas: moderngl.Buffer | None = None
        self._vao_lineas: moderngl.VertexArray | None = None
        self._n_lineas = 0

        self._vbo_tex = self.ctx.buffer(reserve=200 * 4)
        self._vao_tex = self.ctx.vertex_array(self.prog_tex, [(self._vbo_tex, '3f 2f', 'in_position', 'in_uv')])

    def crear_cuad(self, tipo, coord, opacidad, color_base):
        size = 30.0
        r, g, b = color_base
        c = [r, g, b, opacidad]
        
        if tipo == "transversal": # Plano XZ (Y constante)
            y = coord
            v = [
                -size, y, -size, *c,
                 size, y, -size, *c,
                -size, y,  size, *c,
                 size, y, -size, *c,
                 size, y,  size, *c,
                -size, y,  size, *c,
            ]
        elif tipo == "coronal": # Plano XY (Z constante)
            z = coord
            v = [
                -size, 0.0, z, *c,
                 size, 0.0, z, *c,
                -size, 100.0, z, *c,
                 size, 0.0, z, *c,
                 size, 100.0, z, *c,
                -size, 100.0, z, *c,
            ]
        else: # sagital Plano ZY (X constante)
            x = coord
            v = [
                x, 0.0, -size, *c,
                x, 0.0,  size, *c,
                x, 100.0,-size, *c,
                x, 0.0,  size, *c,
                x, 100.0, size, *c,
                x, 100.0,-size, *c,
            ]
        return v

    def crear_grilla(self, tipo, coord, tamano):
        verts = []
        c = [1.0, 1.0, 1.0, 0.4] # Grid visible pero translucida
        
        y_min, y_max = 0.0, 100.0
        x_min, x_max = -30.0, 30.0
        z_min, z_max = -30.0, 30.0
        
        if tipo == "transversal": # Y constante
            y = coord
            for x in np.arange(x_min, x_max + tamano, tamano):
                verts.extend([x, y, z_min, *c,  x, y, z_max, *c])
            for z in np.arange(z_min, z_max + tamano, tamano):
                verts.extend([x_min, y, z, *c,  x_max, y, z, *c])
        elif tipo == "coronal": # Z constante
            z = coord
            for x in np.arange(x_min, x_max + tamano, tamano):
                verts.extend([x, y_min, z, *c,  x, y_max, z, *c])
            for y in np.arange(y_min, y_max + tamano, tamano):
                verts.extend([x_min, y, z, *c,  x_max, y, z, *c])
        elif tipo == "sagital": # X constante
            x = coord
            for z in np.arange(z_min, z_max + tamano, tamano):
                verts.extend([x, y_min, z, *c,  x, y_max, z, *c])
            for y in np.arange(y_min, y_max + tamano, tamano):
                verts.extend([x, y, z_min, *c,  x, y, z_max, *c])
                
        return verts

    def actualizar(self, volumen: VolumenAnatomico):
        verts = []
        vista = volumen.vista_actual
        
        for p in volumen.planos:
            if p.tipo != vista and volumen.esconder_inactivos and vista != "default":
                continue

            opacidad = 0.05 # Base 95% transparente

            color = (1.0, 1.0, 1.0)
            
            if vista == "transversal":
                if p.tipo == "transversal":
                    # Clipping: no dibujar planos por encima del seleccionado
                    if p.coord > volumen.y_nivel + 0.1: continue
                    
                    dist = abs(p.coord - volumen.y_nivel)
                    if dist < 2.5 / volumen.densidad:
                        opacidad = 0.8
                        color = (0.5, 0.9, 0.5)
                    else:
                        opacidad = 0.05
                        color = (0.2, 0.5, 0.2)
                        
            elif vista == "coronal":
                if p.tipo == "coronal":
                    if p.coord > volumen.z_nivel + 0.1: continue
                    
                    dist = abs(p.coord - volumen.z_nivel)
                    if dist < 2.5 / volumen.densidad:
                        opacidad = 0.8
                        color = (0.5, 0.5, 0.9)
                    else:
                        opacidad = 0.05
                        color = (0.2, 0.2, 0.5)
                        
            elif vista == "sagital":
                if p.tipo == "sagital":
                    if p.coord > volumen.x_nivel + 0.1: continue
                    
                    dist = abs(p.coord - volumen.x_nivel)
                    if dist < 2.5 / volumen.densidad:
                        opacidad = 0.8
                        color = (0.9, 0.5, 0.5)
                    else:
                        opacidad = 0.05
                        color = (0.5, 0.2, 0.2)
            else:
                # Default view
                opacidad = 0.08
                if p.tipo == "transversal": color = (0.2, 0.8, 0.2)
                elif p.tipo == "coronal": color = (0.2, 0.2, 0.8)
                else: color = (0.8, 0.2, 0.2)

            if opacidad > 0:
                verts.extend(self.crear_cuad(p.tipo, p.coord, opacidad, color))

        # Renderizar Caras (Triangulares)
        for cara in volumen.caras.values():
            c = [*cara.color, 1.0] # Opaco pero sujeto a test de profundidad
            if len(cara.nodos_ids) >= 3:
                p0 = volumen.nodos[cara.nodos_ids[0]].coord
                for i in range(1, len(cara.nodos_ids) - 1):
                    p1 = volumen.nodos[cara.nodos_ids[i]].coord
                    p2 = volumen.nodos[cara.nodos_ids[i+1]].coord
                    verts.extend([*p0, *c, *p1, *c, *p2, *c])

        # Reconstruir planos
        if verts:
            data = np.array(verts, dtype='f4').tobytes()
            if self._vbo: self._vbo.release()
            if self._vao: self._vao.release()
            self._vbo = self.ctx.buffer(data)
            self._vao = self.ctx.vertex_array(
                self.prog, [(self._vbo, '3f 4f', 'in_position', 'in_color')]
            )
            self._n_verts = len(verts) // 7
        else:
            self._vao = None

        # Reconstruir lineas (Aristas y Temporales)
        lin_verts = []
        
        if vista != "default" and volumen.snap_grid:
            coord = 0.0
            if vista == "transversal": coord = volumen.y_nivel
            elif vista == "coronal": coord = volumen.z_nivel
            elif vista == "sagital": coord = volumen.x_nivel
            lin_verts.extend(self.crear_grilla(vista, coord, volumen.tamano_grid))

        for a in volumen.aristas.values():
            p1 = volumen.nodos[a.n1].coord
            p2 = volumen.nodos[a.n2].coord
            c = [1.0, 1.0, 0.0, 1.0] # Amarillo
            lin_verts.extend([*p1, *c, *p2, *c])
            
        # UI Previsualización Cursor
        if volumen.modo_edicion == "linea" and volumen.nodo_start_linea is not None and volumen.cursor_3d:
            p1 = volumen.nodos[volumen.nodo_start_linea].coord
            p2 = volumen.cursor_3d
            c = [1.0, 0.5, 0.0, 1.0]
            lin_verts.extend([*p1, *c, *p2, *c])
            
        elif volumen.modo_edicion == "cara" and len(volumen.nodos_temp_cara) > 0:
            c = [0.0, 0.8, 1.0, 1.0]
            for i in range(len(volumen.nodos_temp_cara) - 1):
                p1 = volumen.nodos[volumen.nodos_temp_cara[i]].coord
                p2 = volumen.nodos[volumen.nodos_temp_cara[i+1]].coord
                lin_verts.extend([*p1, *c, *p2, *c])
                
            p_last = volumen.nodos[volumen.nodos_temp_cara[-1]].coord
            if volumen.cursor_3d:
                lin_verts.extend([*p_last, *c, *volumen.cursor_3d, *c])
            p_first = volumen.nodos[volumen.nodos_temp_cara[0]].coord
            if len(volumen.nodos_temp_cara) >= 2 and volumen.cursor_3d:
                # Ghost edge
                lin_verts.extend([*volumen.cursor_3d, 0.0, 0.8, 1.0, 0.3, *p_first, 0.0, 0.8, 1.0, 0.3])
                
        # Hit Nodos Draw (Dots)
        for n in volumen.nodos.values():
            c = [1.0, 0.0, 1.0, 0.8]
            sz = 0.2
            lin_verts.extend([n.coord[0]-sz, n.coord[1], n.coord[2], *c, n.coord[0]+sz, n.coord[1], n.coord[2], *c])
            lin_verts.extend([n.coord[0], n.coord[1]-sz, n.coord[2], *c, n.coord[0], n.coord[1]+sz, n.coord[2], *c])
            lin_verts.extend([n.coord[0], n.coord[1], n.coord[2]-sz, *c, n.coord[0], n.coord[1], n.coord[2]+sz, *c])
            
        if lin_verts:
            ldata = np.array(lin_verts, dtype='f4').tobytes()
            if self._vbo_lineas: self._vbo_lineas.release()
            if self._vao_lineas: self._vao_lineas.release()
            self._vbo_lineas = self.ctx.buffer(ldata)
            self._vao_lineas = self.ctx.vertex_array(
                self.prog, [(self._vbo_lineas, '3f 4f', 'in_position', 'in_color')]
            )
            self._n_lineas = len(lin_verts) // 7
        else:
            self._vao_lineas = None
            
        self.update_tex_quad(volumen)

    def update_tex_quad(self, volumen):
        if not volumen.ref_tex: return
        vista, coord, aspect = volumen.ref_vista, volumen.ref_coord, volumen.ref_aspect
        sc = volumen.ref_scale
        u = volumen.ref_offset_u
        v = volumen.ref_offset_v
        
        if vista == "transversal": # Y = coord
            sx = 30.0 * sc; sz = (30.0 / aspect if aspect > 1.0 else 30.0) * sc
            if aspect <= 1.0: sx = 30.0 * aspect * sc
            cx, cz = u, v
            verts = [cx-sx, coord, cz-sz, 0.0, 1.0,  cx+sx, coord, cz-sz, 1.0, 1.0,  cx+sx, coord, cz+sz, 1.0, 0.0,
                     cx-sx, coord, cz-sz, 0.0, 1.0,  cx+sx, coord, cz+sz, 1.0, 0.0,  cx-sx, coord, cz+sz, 0.0, 0.0]
        elif vista == "coronal": # Z = coord
            sx = 30.0 * sc; sy = (30.0 / aspect if aspect > 1.0 else 50.0) * sc
            if aspect <= 1.0: sx = 50.0 * aspect * sc
            cx, cy = u, 50.0 + v
            verts = [cx-sx, cy-sy, coord, 0.0, 1.0,  cx+sx, cy-sy, coord, 1.0, 1.0,  cx+sx, cy+sy, coord, 1.0, 0.0,
                     cx-sx, cy-sy, coord, 0.0, 1.0,  cx+sx, cy+sy, coord, 1.0, 0.0,  cx-sx, cy+sy, coord, 0.0, 0.0]
        else: # sagital X = coord
            sz = 30.0 * sc; sy = (30.0 / aspect if aspect > 1.0 else 50.0) * sc
            if aspect <= 1.0: sz = 50.0 * aspect * sc
            cz, cy = u, 50.0 + v
            verts = [coord, cy-sy, cz-sz, 0.0, 1.0,  coord, cy-sy, cz+sz, 1.0, 1.0,  coord, cy+sy, cz+sz, 1.0, 0.0,
                     coord, cy-sy, cz-sz, 0.0, 1.0,  coord, cy+sy, cz+sz, 1.0, 0.0,  coord, cy+sy, cz-sz, 0.0, 0.0]
        
        self._vbo_tex.write(struct.pack(f'{len(verts)}f', *verts))

    def render(self, mvp: glm.mat4, volumen):
        self.ctx.enable(moderngl.BLEND | moderngl.DEPTH_TEST)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.depth_func = '<='
        self.prog['u_mvp'].write(mvp.to_bytes())
        
        if self._vao:
            self._vao.render(moderngl.TRIANGLES)
            
        if volumen.ref_tex:
            volumen.ref_tex.use(0)
            self.prog_tex['Tex'].value = 0
            self.prog_tex['u_mvp'].write(mvp.to_bytes())
            self.prog_tex['u_opacity'].value = volumen.ref_opacity
            self._vao_tex.render(moderngl.TRIANGLES)

        if self._vao_lineas:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self._vao_lineas.render(moderngl.LINES)
            self.ctx.enable(moderngl.DEPTH_TEST)

# ════════════════════════════════════════════════════════════
# CÁMARA ORBITAL
# ════════════════════════════════════════════════════════════

class CamaraOrbital:
    def __init__(self):
        self.centro = glm.vec3(0.0, 50.0, 0.0)
        self.distancia = 120.0
        self.yaw = math.radians(45.0)
        self.pitch = math.radians(20.0)

    def matriz_vista(self) -> glm.mat4:
        cx = self.distancia * math.cos(self.pitch) * math.sin(self.yaw)
        cy = self.distancia * math.sin(self.pitch)
        cz = self.distancia * math.cos(self.pitch) * math.cos(self.yaw)
        pos = self.centro + glm.vec3(cx, cy, cz)
        return glm.lookAt(pos, self.centro, glm.vec3(0, 1, 0))

    def matriz_proyeccion(self, w: int, h: int) -> glm.mat4:
        aspect = w / h if h > 0 else 1.0
        return glm.perspective(glm.radians(45.0), aspect, 0.1, 1000.0)

# ════════════════════════════════════════════════════════════
# APLICACIÓN
# ════════════════════════════════════════════════════════════

class App:
    def __init__(self):
        pygame.init()
        # Ventana inicial normal, delegada a Maximize para respetar Taskbars/Bordes
        self.w, self.h = 1280, 720
        
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE | pygame.WINDOWMAXIMIZED
        pygame.display.set_mode((self.w, self.h), flags)
        
        pygame.display.set_caption("Láminas Volumétricas Anatómicas")

        self.ctx = moderngl.create_context()
        
        if HAS_IMGUI:
            imgui.create_context()
            self.imgui_renderer = PygameRenderer()
            imgui.get_io().display_size = (float(self.w), float(self.h))

        self.volumen = VolumenAnatomico()
        self.camara = CamaraOrbital()
        self.renderer = Renderer(self.ctx)

        self._rebuild_gpu()
        
        self._drag_rotate = False
        self.clock = pygame.time.Clock()
        self.ejecutando = True

        self._mostrar_modal_salir = False
        self.mostrar_propiedades_imagen = False

    def centrar_camara_plano(self):
        v = self.volumen.vista_actual
        if v == "transversal":
            self.camara.centro = glm.vec3(0.0, self.volumen.y_nivel, 0.0)
        elif v == "coronal":
            self.camara.centro = glm.vec3(0.0, 50.0, self.volumen.z_nivel)
        elif v == "sagital":
            self.camara.centro = glm.vec3(self.volumen.x_nivel, 50.0, 0.0)
        self.camara.distancia = 80.0

    def check_colision_textura_lclick(self):
        vol = self.volumen
        if not vol.ref_tex: return False
        
        # Proyectar usando el plano de la imagen específicamente
        res = unproject(*pygame.mouse.get_pos(), self.w, self.h, self.camara, vol.ref_vista, vol.ref_coord, vol.ref_coord, vol.ref_coord)
        if not res: return False
        
        aspect = vol.ref_aspect
        sc = vol.ref_scale
        u = vol.ref_offset_u
        v = vol.ref_offset_v
        
        if vol.ref_vista == "transversal":
            cx, cz = u, v
            sx = 30.0 * sc; sz = (30.0 / aspect if aspect > 1.0 else 30.0) * sc
            if aspect <= 1.0: sx = 30.0 * aspect * sc
            return abs(res[0]-cx) <= sx and abs(res[2]-cz) <= sz
            
        elif vol.ref_vista == "coronal":
            cx, cy = u, 50.0 + v
            sx = 30.0 * sc; sy = (30.0 / aspect if aspect > 1.0 else 50.0) * sc
            if aspect <= 1.0: sx = 50.0 * aspect * sc
            return abs(res[0]-cx) <= sx and abs(res[1]-cy) <= sy
            
        else: # sagital
            cz, cy = u, 50.0 + v
            sz = 30.0 * sc; sy = (30.0 / aspect if aspect > 1.0 else 50.0) * sc
            if aspect <= 1.0: sz = 50.0 * aspect * sc
            return abs(res[2]-cz) <= sz and abs(res[1]-cy) <= sy

    def _rebuild_gpu(self):
        # We must sort back-to-front by camera distance for proper alpha blending
        # But for simple visualization, we rely on depth test + blending
        self.renderer.actualizar(self.volumen)
        
    def _snap(self, p):
        if not self.volumen.snap_grid: return p
        tg = self.volumen.tamano_grid
        return (round(p[0]/tg)*tg, round(p[1]/tg)*tg, round(p[2]/tg)*tg)

    def run(self):
        while self.ejecutando:
            self.ctx.clear(0.05, 0.05, 0.08)
            self._handle_events()
            
            if HAS_IMGUI: self._draw_imgui()

            mvp = self.camara.matriz_proyeccion(self.w, self.h) * self.camara.matriz_vista()
            self.renderer.render(mvp, self.volumen)
            if HAS_IMGUI:
                imgui.render()
                self.imgui_renderer.render(imgui.get_draw_data())
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

    def _handle_events(self):
        teclas = pygame.key.get_pressed()
        vel_cam = 1.0
        v = self.volumen.vista_actual
        
        if v == "transversal":
            if teclas[pygame.K_w]: self.camara.centro.z -= vel_cam
            if teclas[pygame.K_s]: self.camara.centro.z += vel_cam
            if teclas[pygame.K_a]: self.camara.centro.x -= vel_cam
            if teclas[pygame.K_d]: self.camara.centro.x += vel_cam
        elif v == "coronal":
            if teclas[pygame.K_w]: self.camara.centro.y += vel_cam
            if teclas[pygame.K_s]: self.camara.centro.y -= vel_cam
            if teclas[pygame.K_a]: self.camara.centro.x -= vel_cam
            if teclas[pygame.K_d]: self.camara.centro.x += vel_cam
        elif v == "sagital":
            if teclas[pygame.K_w]: self.camara.centro.y += vel_cam
            if teclas[pygame.K_s]: self.camara.centro.y -= vel_cam
            if teclas[pygame.K_a]: self.camara.centro.z -= vel_cam
            if teclas[pygame.K_d]: self.camara.centro.z += vel_cam
        else:
            if teclas[pygame.K_w]:
                self.camara.centro.x -= math.sin(self.camara.yaw) * vel_cam
                self.camara.centro.z -= math.cos(self.camara.yaw) * vel_cam
            if teclas[pygame.K_s]:
                self.camara.centro.x += math.sin(self.camara.yaw) * vel_cam
                self.camara.centro.z += math.cos(self.camara.yaw) * vel_cam
            if teclas[pygame.K_a]:
                self.camara.centro.x -= math.cos(self.camara.yaw) * vel_cam
                self.camara.centro.z += math.sin(self.camara.yaw) * vel_cam
            if teclas[pygame.K_d]:
                self.camara.centro.x += math.cos(self.camara.yaw) * vel_cam
                self.camara.centro.z -= math.sin(self.camara.yaw) * vel_cam
            if teclas[pygame.K_e]:
                self.camara.centro.y += vel_cam
            if teclas[pygame.K_q]:
                self.camara.centro.y -= vel_cam

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                self._mostrar_modal_salir = True
            elif evento.type == pygame.VIDEORESIZE:
                self.w, self.h = evento.w, evento.h
                self.ctx.viewport = (0, 0, self.w, self.h)
                if HAS_IMGUI: imgui.get_io().display_size = (float(self.w), float(self.h))
            
            if HAS_IMGUI:
                self.imgui_renderer.process_event(evento)
                if imgui.get_io().want_capture_mouse:
                    continue

            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.volumen.undo()
                    self._rebuild_gpu()
                elif evento.key == pygame.K_c:
                    self.centrar_camara_plano()
            
            if evento.type == pygame.MOUSEBUTTONDOWN:
                if evento.button == 1:
                    if self.check_colision_textura_lclick():
                        self.mostrar_propiedades_imagen = not self.mostrar_propiedades_imagen
                    else:
                        self._drag_rotate = True
                elif evento.button == 3:
                    if self.volumen.vista_actual != "default":
                        res = unproject(*pygame.mouse.get_pos(), self.w, self.h, self.camara,
                                        self.volumen.vista_actual, self.volumen.y_nivel, self.volumen.z_nivel, self.volumen.x_nivel)
                        if res: res = self._snap(res)
                        if res:
                            nid, is_new = self.volumen.add_nodo(res)
                            if self.volumen.modo_edicion == "linea":
                                if self.volumen.nodo_start_linea is None:
                                    self.volumen.nodo_start_linea = nid
                                else:
                                    self.volumen.add_arista(self.volumen.nodo_start_linea, nid)
                                    self.volumen.nodo_start_linea = None
                            elif self.volumen.modo_edicion == "cara":
                                if self.volumen.nodos_temp_cara and self.volumen.nodos_temp_cara[0] == nid and len(self.volumen.nodos_temp_cara) >= 3:
                                    # Cerrar loop y consolidar cara
                                    self.volumen.add_cara(self.volumen.nodos_temp_cara.copy())
                                    self.volumen.nodos_temp_cara.clear()
                                else:
                                    # Evitar dobles clicks adyacentes
                                    if not self.volumen.nodos_temp_cara or self.volumen.nodos_temp_cara[-1] != nid:
                                        self.volumen.nodos_temp_cara.append(nid)
                            self._rebuild_gpu()
                                    
            elif evento.type == pygame.MOUSEBUTTONUP:
                if evento.button == 1: self._drag_rotate = False
                elif evento.button == 3: pass
                         
            elif evento.type == pygame.MOUSEWHEEL:
                self.camara.distancia -= evento.y * 5.0
                self.camara.distancia = max(10.0, min(self.camara.distancia, 300.0))
                
            elif evento.type == pygame.MOUSEMOTION:
                res = unproject(*pygame.mouse.get_pos(), self.w, self.h, self.camara,
                                self.volumen.vista_actual, self.volumen.y_nivel, self.volumen.z_nivel, self.volumen.x_nivel)
                if res and self.volumen.snap_grid: res = self._snap(res)
                self.volumen.cursor_3d = res
                
                if self._drag_rotate:
                    dx, dy = evento.rel
                    self.camara.yaw -= dx * 0.005
                    self.camara.pitch += dy * 0.005
                    self.camara.pitch = max(-1.5, min(1.5, self.camara.pitch))
                    
                self._rebuild_gpu()

    def _draw_imgui(self):
        if BUNDLE: imgui.new_frame()
        else: imgui.new_frame()
        
        # ── INTERCEPTOR OVERLAY SALIDA ──
        if self._mostrar_modal_salir:
            imgui.open_popup("¿Salir sin guardar?")
            
        modal_flags = imgui.WindowFlags_.always_auto_resize if BUNDLE else imgui.WINDOW_ALWAYS_AUTO_RESIZE
        if imgui.begin_popup_modal("¿Salir sin guardar?", flags=modal_flags)[0]:
            imgui.text("Estás a punto de salir.")
            imgui.text("Los cambios que no hayas mapeado o guardado se perderán.")
            imgui.separator()
            if imgui.button("Sí, Salir", 120, 30 if BUNDLE else 0):
                self.ejecutando = False
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancelar", 120, 30 if BUNDLE else 0):
                self._mostrar_modal_salir = False
                imgui.close_current_popup()
            imgui.end_popup()

        # ── Barra de Menú Principal (AutoCAD Style) ──
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Archivo"):
                imgui.menu_item("Nuevo proyecto", "", False)
                
                if imgui.menu_item("Abrir imagen de referencia", "", False)[0]:
                    root = tk.Tk()
                    root.withdraw()
                    ruta = filedialog.askopenfilename(title="Seleccionar Referencia", filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg")])
                    root.destroy()
                    if ruta:
                        try:
                            img = pygame.image.load(ruta).convert_alpha()
                            w, h = img.get_size()
                            tex_data = pygame.image.tostring(img, "RGBA")
                            
                            if self.volumen.ref_tex:
                                self.volumen.ref_tex.release()
                                
                            self.volumen.ref_tex = self.ctx.texture((w, h), 4, tex_data)
                            self.volumen.ref_aspect = w / h
                            self.volumen.ref_scale = 1.0
                            self.volumen.ref_offset_u = 0.0
                            self.volumen.ref_offset_v = 0.0
                            self.mostrar_propiedades_imagen = True
                            self._rebuild_gpu()
                        except Exception as e:
                            print("Error cargando textura:", e)

                imgui.menu_item("Guardar (JSON)", "", False)
                imgui.menu_item("Importar texturas", "", False)
                imgui.end_menu()
                
            if imgui.begin_menu("Editar"):
                if imgui.menu_item("Trazado: Conectar Nodos (Aristas)", "", self.volumen.modo_edicion == "linea")[0]:
                    self.volumen.modo_edicion = "linea"
                    self.volumen.nodo_start_linea = None
                    self.volumen.nodos_temp_cara.clear()
                if imgui.menu_item("Trazado: Construir Cara (Polígono)", "", self.volumen.modo_edicion == "cara")[0]:
                    self.volumen.modo_edicion = "cara"
                    self.volumen.nodo_start_linea = None
                    self.volumen.nodos_temp_cara.clear()
                    
                imgui.separator()
                
                ch_r, nf_color = imgui.color_edit3("Color Activo de Cara", self.volumen.color_actual)
                if ch_r: self.volumen.color_actual = list(nf_color)
                
                if imgui.menu_item("Deshacer última acción (Ctrl+Z)", "", False)[0]:
                    self.volumen.undo()
                    self._rebuild_gpu()
                    
                imgui.end_menu()
                
            if imgui.begin_menu("Ver"):
                vistas = [("Transversal", "transversal"), ("Coronal", "coronal"), ("Sagital", "sagital"), ("Combinado", "default")]
                for lbl, v in vistas:
                    if imgui.menu_item(f"Vista: {lbl}", "", self.volumen.vista_actual==v)[0]:
                        self.volumen.set_vista(v)
                        self._rebuild_gpu()
                        
                imgui.separator()
                if imgui.menu_item("Centrar Cámara en Plano (Atajo: C)", "", False)[0]:
                    self.centrar_camara_plano()
                    
                imgui.separator()
                ch_hide, val_hide = imgui.checkbox("Esconder Planos Inactivos", self.volumen.esconder_inactivos)
                if ch_hide:
                    self.volumen.esconder_inactivos = val_hide
                    self._rebuild_gpu()
                imgui.end_menu()
                
            if imgui.begin_menu("Herramientas"):
                if imgui.menu_item("Calcar imagen en plano seleccionado", "", False)[0]:
                    if self.volumen.ref_tex and self.volumen.vista_actual != "default":
                        self.volumen.ref_vista = self.volumen.vista_actual
                        if self.volumen.vista_actual == "transversal": self.volumen.ref_coord = self.volumen.y_nivel
                        elif self.volumen.vista_actual == "coronal": self.volumen.ref_coord = self.volumen.z_nivel
                        elif self.volumen.vista_actual == "sagital": self.volumen.ref_coord = self.volumen.x_nivel
                        self._rebuild_gpu()
                        
                if imgui.menu_item("Quitar imagen de referencia", "", False)[0]:
                    if self.volumen.ref_tex:
                        self.volumen.ref_tex.release()
                        self.volumen.ref_tex = None
                        self.mostrar_propiedades_imagen = False
                        self._rebuild_gpu()
                        
                if imgui.menu_item("Propiedades de Imagen", "", self.mostrar_propiedades_imagen)[0]:
                    self.mostrar_propiedades_imagen = not self.mostrar_propiedades_imagen
                        
                imgui.menu_item("Agrupar regiones por color", "", False)
                
                ch_snap, val_snap = imgui.checkbox("Snap a Grilla Activo", self.volumen.snap_grid)
                if ch_snap:
                    self.volumen.snap_grid = val_snap
                    self._rebuild_gpu()
                imgui.end_menu()
                
            if imgui.begin_menu("Ayuda"):
                imgui.text("LClick: Rotar | RClick: Dibujar")
                imgui.text("WASD+QE: Mover cámara libremente")
                imgui.text("Scroll: Acercar / Alejar")
                imgui.end_menu()
                
            imgui.end_main_menu_bar()

        # ── Propiedades de Imagen (Flotante) ──
        if self.mostrar_propiedades_imagen and self.volumen.ref_tex:
            if BUNDLE: imgui.set_next_window_pos((50, 50), imgui.Cond_.first_use_ever)
            else: imgui.set_next_window_position(50, 50, imgui.FIRST_USE_EVER)
            
            is_open, self.mostrar_propiedades_imagen = imgui.begin("Edición de Imagen", self.mostrar_propiedades_imagen)
            if is_open:
                ch_op, n_op = imgui.slider_float("Transparencia", self.volumen.ref_opacity, 0.0, 1.0)
                if ch_op: self.volumen.ref_opacity = n_op; self._rebuild_gpu()
                
                ch_sc, n_sc = imgui.slider_float("Escala", self.volumen.ref_scale, 0.1, 5.0)
                if ch_sc: self.volumen.ref_scale = n_sc; self._rebuild_gpu()
                
                ch_u, n_u = imgui.slider_float("Mover Horizontal", self.volumen.ref_offset_u, -40.0, 40.0)
                if ch_u: self.volumen.ref_offset_u = n_u; self._rebuild_gpu()
                
                ch_v, n_v = imgui.slider_float("Mover Vertical", self.volumen.ref_offset_v, -60.0, 60.0)
                if ch_v: self.volumen.ref_offset_v = n_v; self._rebuild_gpu()
                
                if imgui.button("Centrar Imagen"):
                    self.volumen.ref_scale = 1.0
                    self.volumen.ref_offset_u = 0.0
                    self.volumen.ref_offset_v = 0.0
                    self._rebuild_gpu()
            imgui.end()

        # ── Panel Secundario Derecho (Propiedades Base) ──
        if BUNDLE:
            imgui.set_next_window_pos((self.w - 320, 30), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size((300, 450), imgui.Cond_.first_use_ever)
        else:
            imgui.set_next_window_position(self.w - 320, 30, imgui.FIRST_USE_EVER)
            imgui.set_next_window_size(300, 450, imgui.FIRST_USE_EVER)
            
        is_expand, _ = imgui.begin("Propiedades y Guias")
        if is_expand:
            imgui.text("Variables Focales:")
            imgui.separator()
            if self.volumen.vista_actual == "transversal":
                c, v = imgui.slider_float("Nivel Y", self.volumen.y_nivel, 0.0, 100.0)
                if c: self.volumen.y_nivel = v; self._rebuild_gpu()
            elif self.volumen.vista_actual == "coronal":
                c, v = imgui.slider_float("Nivel Z", self.volumen.z_nivel, -30.0, 30.0)
                if c: self.volumen.z_nivel = v; self._rebuild_gpu()
            elif self.volumen.vista_actual == "sagital":
                c, v = imgui.slider_float("Nivel X", self.volumen.x_nivel, -30.0, 30.0)
                if c: self.volumen.x_nivel = v; self._rebuild_gpu()
                
            imgui.separator()
            if self.volumen.snap_grid:
                c_tg, v_tg = imgui.slider_float("Tamaño Grid", self.volumen.tamano_grid, 0.5, 5.0)
                if c_tg: self.volumen.tamano_grid = v_tg; self._rebuild_gpu()
                
            imgui.text("Multiplicador Resolutivo:")
            for mul in [1, 2, 3]:
                if mul > 1: imgui.same_line()
                act = self.volumen.densidad == mul
                if imgui.button(f"[{mul}x]" if act else f"{mul}x"):
                    self.volumen.densidad = mul
                    self.volumen.generar_planos()
                    self._rebuild_gpu()
        imgui.end()

        # ── Barra de Status Inferior ──
        if BUNDLE:
            flags = imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_saved_settings
            imgui.set_next_window_pos((0, self.h - 35))
            imgui.set_next_window_size((self.w, 35))
        else:
            flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS
            imgui.set_next_window_position(0, self.h - 35)
            imgui.set_next_window_size(self.w, 35)
        
        is_stat, _ = imgui.begin("Status", flags=flags)
        if is_stat:
            hm = "Construir Aristas" if self.volumen.modo_edicion == "linea" else "Construir Caras"
            msg = f"HERRAMIENTA: {hm} | SNAP: {'ACTIVO' if self.volumen.snap_grid else 'LIBRE'} | NODOS RAM: {len(self.volumen.nodos)} | ARISTAS RAM: {len(self.volumen.aristas)} | CARAS: {len(self.volumen.caras)}"
            if BUNDLE: imgui.text_colored((0.5, 0.9, 1.0, 1.0), msg)
            else: imgui.text_colored(msg, 0.5, 0.9, 1.0, 1.0)
        imgui.end()

        if BUNDLE: imgui.render()

if __name__ == "__main__":
    app = App()
    app.run()