import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

"""
从 editor3dJson.json 解析 modelInstances 并生成 MJCF 模型

数据结构：
- modelInstances: 模型实例列表
  - uniqueId: 唯一标识
  - position: {x, y, z} 位置
  - rotate: {x, y, z} 旋转（弧度）
  - meshes: mesh 数据列表//~
    - coordinateArray: 顶点坐标 [x1,y1,z1,x2,y2,z2,...]
    - coordinateIndexArray: 面索引 [i1,i2,i3,...]
"""

def parse_editor3d_json(json_path):
    """
    解析 editor3d JSON 文件
    
    返回:
        instances: 列表，每个元素包含:
          - unique_id: 唯一标识
          - position: (x, y, z) 位置
          - rotation: (x, y, z) 旋转
          - meshes: mesh 数据列表 [(vertices, faces), ...]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'modelInstances' not in data:
        print("❌ 错误: JSON 文件中没有找到 'modelInstances' 字段")
        return []
    
    instances = []
    
    for idx, instance in enumerate(data['modelInstances']):
        unique_id = instance.get('uniqueId', f'instance_{idx}')
        
        # 提取位置
        pos_dict = instance.get('position', {})
        position = (
            pos_dict.get('x', 0.0),
            pos_dict.get('y', 0.0),
            pos_dict.get('z', 0.0)
        )
        
        # 提取旋转（弧度）
        rot_dict = instance.get('rotate', {})
        rotation = (
            rot_dict.get('x', 0.0),
            rot_dict.get('y', 0.0),
            rot_dict.get('z', 0.0)
        )
        
        # 提取 meshes
        meshes_data = []
        for mesh_idx, mesh in enumerate(instance.get('meshes', [])):
            if 'coordinateArray' in mesh and 'coordinateIndexArray' in mesh:
                vertices = np.array(mesh['coordinateArray'], dtype=np.float64)
                faces = np.array(mesh['coordinateIndexArray'], dtype=np.int32)
                
                num_vertices = len(vertices) // 3
                num_faces = len(faces) // 3
                
                meshes_data.append({
                    'vertices': vertices,
                    'faces': faces,
                    'num_vertices': num_vertices,
                    'num_faces': num_faces,
                    'mesh_index': mesh_idx
                })
        
        if meshes_data:  # 只添加有 mesh 数据的实例
            instances.append({
                'unique_id': unique_id,
                'position': position,
                'rotation': rotation,
                'meshes': meshes_data,
                'instance_index': idx
            })
            
            print(f"实例 {idx} ({unique_id}):")
            print(f"  位置: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
            print(f"  旋转: ({rotation[0]:.2f}, {rotation[1]:.2f}, {rotation[2]:.2f}) rad")
            print(f"  Mesh 数量: {len(meshes_data)}")
            for mesh in meshes_data:
                print(f"    Mesh {mesh['mesh_index']}: {mesh['num_vertices']} 顶点, {mesh['num_faces']} 面")
    
    return instances

def rotation_to_quaternion(rx, ry, rz):
    """
    将 XYZ 欧拉角（弧度）转换为四元数 (w, x, y, z)
    """
    r = Rotation.from_euler('xyz', [rx, ry, rz], degrees=False)
    quat = r.as_quat()  # 返回 (x, y, z, w)
    return [quat[3], quat[0], quat[1], quat[2]]  # 转换为 (w, x, y, z)

# ============================================
# 解析 JSON 文件
# ============================================

print("=" * 60)
print("解析 editor3dJson.json")
print("=" * 60)

instances = parse_editor3d_json("editor3dJson.json")

if not instances:
    print("\n❌ 没有找到有效的模型实例")
    exit(1)

print(f"\n✅ 成功解析 {len(instances)} 个模型实例\n")

# ============================================
# 创建 MuJoCo 模型
# ============================================

spec = mujoco.MjSpec()
spec.modelname = "editor3d_model"
spec.compiler.degree = False
spec.compiler.inertiafromgeom = True
spec.option.gravity = [0, 0, -9.81]
spec.option.timestep = 0.002

# 坐标缩放因子（根据数据单位调整）
scale_factor = 0.001  # 假设原始单位是毫米

print("=" * 60)
print("创建 MuJoCo 模型")
print("=" * 60)

# 为每个实例创建 mesh 和 body
for inst in instances:
    unique_id = inst['unique_id']
    position = inst['position']
    rotation = inst['rotation']
    
    # 应用缩放
    pos_scaled = [p * scale_factor for p in position]
    
    # 转换旋转为四元数
    quat = rotation_to_quaternion(rotation[0], rotation[1], rotation[2])
    
    print(f"\n处理实例: {unique_id}")
    
    # 为该实例的每个 mesh 创建 mesh 资源
    for mesh_data in inst['meshes']:
        mesh_idx = mesh_data['mesh_index']
        mesh_name = f"{unique_id}_mesh_{mesh_idx}"
        
        # 创建 mesh
        mesh = spec.add_mesh()
        mesh.name = mesh_name
        mesh.uservert = mesh_data['vertices']
        mesh.userface = mesh_data['faces']
        mesh.scale = [scale_factor, scale_factor, scale_factor]
        
        print(f"  创建 mesh: {mesh_name} ({mesh_data['num_vertices']} 顶点, {mesh_data['num_faces']} 面)")
    
    # 创建 body（如果有多个 mesh，创建一个包含多个 geom 的 body）
    body = spec.worldbody.add_body()
    body.name = f"body_{unique_id}"
    body.pos = pos_scaled
    body.quat = quat
    
    # 为每个 mesh 创建 geom
    for mesh_data in inst['meshes']:
        mesh_idx = mesh_data['mesh_index']
        mesh_name = f"{unique_id}_mesh_{mesh_idx}"
        
        geom = body.add_geom()
        geom.name = f"geom_{unique_id}_{mesh_idx}"
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh_name
        geom.rgba = [0.7, 0.7, 0.7, 1.0]  # 默认灰色
        geom.mass = 1.0
        geom.contype = 1
        geom.conaffinity = 1
    
    print(f"  创建 body: body_{unique_id}")
    print(f"  位置: ({pos_scaled[0]:.3f}, {pos_scaled[1]:.3f}, {pos_scaled[2]:.3f})")
    print(f"  四元数: ({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")

# 添加地面
ground = spec.worldbody.add_body()
ground.name = "ground"
ground.pos = [0, 0, -0.1]

ground_geom = ground.add_geom()
ground_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
ground_geom.size = [5, 5, 0.1]
ground_geom.rgba = [0.8, 0.8, 0.8, 1]

print(f"\n✅ 创建地面")

# ============================================
# 编译并保存
# ============================================

print("\n" + "=" * 60)
print("编译模型")
print("=" * 60)

try:
    model = spec.compile()
    print("✅ 模型编译成功！")
    
    xml_string = spec.to_xml()
    
    with open("editor3d_model.xml", "w") as f:
        f.write(xml_string)
    
    print("✅ MJCF 文件已保存: editor3d_model.xml")
    
    # 生成简化的查看版本（前100行）
    lines = xml_string.split('\n')
    if len(lines) > 100:
        preview = '\n'.join(lines[:100]) + '\n...\n(共 {} 行)'.format(len(lines))
    else:
        preview = xml_string
    
    print("\n模型预览:")
    print("-" * 60)
    print(preview)
    
    # 统计信息
    print("\n" + "=" * 60)
    print("模型统计")
    print("=" * 60)
    print(f"Body 数量: {model.nbody}")
    print(f"Geom 数量: {model.ngeom}")
    print(f"总顶点数: {sum(mesh['num_vertices'] for inst in instances for mesh in inst['meshes'])}")
    print(f"总面数: {sum(mesh['num_faces'] for inst in instances for mesh in inst['meshes'])}")
    
    print("\n✅ 完成！所有 modelInstances 已转换为 MJCF 模型")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

