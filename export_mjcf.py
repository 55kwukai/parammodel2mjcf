import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from collections import defaultdict
import os
from typing import List, Dict, Set, Tuple, Optional, Any
import parse_editor3d_json




# 字段映射配置：原字段名 -> 新字段名
PROPERTY_MAPPING = {
    "hkglwt": "parent_body",       # 滑块关联模型 -> 父物体
    "hgzzmx": "current_body",      # 滑轨终止模型 -> 当前物体
    "hingeActuator": "hinge_actuator", # 保持原名
    "mdfwtest": "actuator_range"      # 马达范围 (可选优化)
}

def normalize_auxiliaries(json_path: str) -> List[Dict]:
    """
    读取 JSON 文件并解析辅助结构数据。

    功能：
    1. 读取并验证 JSON 文件。
    2. 提取核心变换数据 (ID, Type, Position, Rotation)。
    3. 扁平化 paramBizProperties 列表。
    4. 根据 PROPERTY_MAPPING 重命名业务属性，使其更具可读性。
    """

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件未找到: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 格式错误: {e}")

    # 兼容直接列表或包含在 customAuxiliaries 对象中的结构
    raw_list = data.get("customAuxiliaries", []) if isinstance(data, dict) else data
    if not isinstance(raw_list, list):
        return []

    normalized_list = []

    for item in raw_list:
        # 1. 提取基础变换信息
        pos = item.get("position", {})
        rot = item.get("rotate", {})

        # 构建基础对象结构
        aux_obj = {
            "id": item.get("instanceId"),
            "joint_type": item.get("auxiliaryType"),
            "position": [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)],
            "rotation": [rot.get("x", 0), rot.get("y", 0), rot.get("z", 0)]
        }

        # 2. 处理业务属性 (paramBizProperties)
        # 将列表转换为扁平化的键值对，并执行重命名
        raw_props = item.get("paramBizProperties", [])

        for prop in raw_props:
            original_key = prop.get("key")
            raw_value = prop.get("value")
            val_type = prop.get("valueType")

            # 值类型转换
            if val_type == "boolean":
                final_value = True if str(raw_value).lower() == "true" else False
            else:
                final_value = raw_value

            # 键名映射与赋值
            # 如果 key 在映射表中，使用新名字；否则保留原名（或根据需求选择忽略）
            target_key = PROPERTY_MAPPING.get(original_key, original_key)

            aux_obj[target_key] = final_value

        normalized_list.append(aux_obj)

    return normalized_list


def print_auxiliaries(auxiliaries: List[Dict]):
    """
    以表格形式打印处理后的辅助结构数据
    """
    if not auxiliaries:
        print("列表为空。")
        return

    # 定义列宽格式
    row_fmt = "{:<5} | {:<5} | {:<20} | {:<10} | {:<20} | {:<20}"
    header = row_fmt.format("ID", "Type", "Position", "Actuator", "Parent Body", "Current Body")

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for aux in auxiliaries:
        # 格式化坐标显示
        pos_str = f"[{aux['position'][0]}, {aux['position'][1]}, {aux['position'][2]}]"

        # 获取属性，提供默认值以防缺失
        has_motor = "Yes" if aux.get("hinge_actuator") else "No"
        p_body = str(aux.get("parent_body", "N/A"))
        c_body = str(aux.get("current_body", "N/A"))

        print(row_fmt.format(
            aux['id'],
            aux['joint_type'],
            pos_str,
            has_motor,
            p_body,
            c_body
        ))

    print("-" * len(header))



def build_topology_tree(auxiliaries: List[Dict]) -> List[Dict]:
    """
    根据 auxiliaries 列表构建拓扑树。

    返回:
        一个列表，包含所有的根节点（可能有多个独立的树/森林）。
        每个节点结构如下:
        {
            "body_id": str,          # 物体 ID
            "children": List[Dict],  # 子节点列表
            "link_data": Dict        # 连接到该节点的辅助体数据 (作为子节点时才有，根节点为 None)
        }
    """
    if not auxiliaries:
        return []

    # 1. 建立邻接表 (Adjacency List) 和 子节点集合
    # 结构: parent_id -> [ (child_id, auxiliary_data), ... ]
    adj_list = {}
    all_bodies = set()
    children_bodies = set()

    for aux in auxiliaries:
        # 获取父子 ID，转换为字符串以保证一致性
        # 如果 parent_body 为空或 N/A，通常视为连接到 World (根)
        p_body = str(aux.get("parent_body")) if aux.get("parent_body") else "World"
        c_body = str(aux.get("current_body"))

        # 记录出现过的所有 Body
        all_bodies.add(p_body)
        all_bodies.add(c_body)

        # 记录这是一个子节点
        children_bodies.add(c_body)

        # 填充邻接表
        if p_body not in adj_list:
            adj_list[p_body] = []

        adj_list[p_body].append({
            "child_id": c_body,
            "aux_data": aux
        })

    # 2. 寻找根节点 (Roots)
    # 根节点是那些存在于 all_bodies 中，但从未作为 child 出现的 body
    roots = list(all_bodies - children_bodies)

    # 如果数据只有环路没有根（极少见），或者为空，做个保护
    if not roots and all_bodies:
        # 这里的逻辑视具体业务而定，通常如果没有根，可能意味着这是一个完全闭环或者数据缺失
        # 这里为了演示，假设第一个出现的 parent 为根
        roots = [list(adj_list.keys())[0]]

    # 3. 递归构建树函数
    def _build_node(body_id: str, link_data: Dict = None) -> Dict:
        node = {
            "body_id": body_id,
            "link_data": link_data,  # 记录是哪个 auxiliary 连接到这个 body 的
            "children": []
        }

        # 查找该 body 下挂载的所有子 body
        if body_id in adj_list:
            for item in adj_list[body_id]:
                child_node = _build_node(item["child_id"], item["aux_data"])
                node["children"].append(child_node)

        return node

    # 4. 构建所有根节点的树
    forest = []
    for root_id in roots:
        forest.append(_build_node(root_id))

    return forest


def print_topology_tree(trees: List[Dict]):
    """
    以图形化树状结构打印拓扑关系。
    """
    if not trees:
        print("拓扑树为空。")
        return

    print("=" * 40)
    print("PHYSICS TOPOLOGY TREE")
    print("=" * 40)

    def _print_node(node: Dict, prefix: str = "", is_last: bool = True, is_root: bool = False):
        body_id = node["body_id"]
        children = node["children"]
        link = node["link_data"]

        # 1. 构造当前行的显示内容
        connector = ""
        if not is_root:
            connector = "└── " if is_last else "├── "

        # 2. 提取连接信息 (如果不是根节点)
        info_str = f"[{body_id}]"
        if link:
            motor = " (Motor)" if link.get('hinge_actuator') else ""

            # 显示格式: └── [JointType: ID] -> [BodyID] (Motor?)
            display_str = f"{prefix}{connector} -> \033[1m{info_str}\033[0m{motor}"
        else:
            # 根节点显示
            display_str = f"{prefix}{info_str} (Root)"

        print(display_str)

        # 3. 计算子节点的前缀
        if is_root:
            new_prefix = prefix
        else:
            new_prefix = prefix + ("    " if is_last else "│   ")

        # 4. 递归打印子节点
        count = len(children)
        for i, child in enumerate(children):
            _print_node(child, new_prefix, i == count - 1, is_root=False)

    # 遍历打印所有的树（通常只有一个，但可能有多个独立的结构）
    for tree in trees:
        _print_node(tree, is_root=True)
        print("-" * 20)  # 分隔符


class MujocoBuilderWithMesh:
    def __init__(self, model_name: str = "robot", asset_dir: str = "assets"):
        self.asset_dir = asset_dir
        self.model_name = model_name

        # 1. 初始化 Spec
        self.spec = mujoco.MjSpec()
        self.spec.modelname = model_name

        # 2. 基础设置
        self.spec.compiler.degree = False
        self.spec.compiler.inertiafromgeom = True
        self.spec.option.gravity = [0, 0, -9.81]
        self.spec.option.timestep = 0.002

        # 3. 状态追踪
        self.processed_ids: Set[str] = set()
        self.instance_map: Dict[str, Dict] = {}

        # 确保资源目录存在
        if not os.path.exists(self.asset_dir):
            os.makedirs(self.asset_dir)

    def _write_obj_file(self, filename: str, vertices: np.ndarray, faces: np.ndarray):
        filepath = os.path.join(self.asset_dir, filename)

        # [修复] 如果是一维数组，重塑为 (N, 3)
        if vertices.ndim == 1:
            vertices = vertices.reshape(-1, 3)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Generated for {self.model_name}\n")
            # 写入顶点
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # 写入面 (索引从1开始)
            num_faces = len(faces) // 3
            for i in range(num_faces):
                idx = i * 3
                f.write(f"f {faces[idx] + 1} {faces[idx + 1] + 1} {faces[idx + 2] + 1}\n")
        return filename

    def _prepare_assets(self, instances: List[Dict]):
        """
        1. 建立 ID 映射
        2. 将 Mesh 数据写入文件
        3. 在 Spec 中注册 Mesh 资源
        """
        print(f"正在处理 {len(instances)} 个实例的资源...")

        for inst in instances:
            uid = inst['unique_id']
            self.instance_map[uid] = inst

            # 处理该实例包含的所有 Sub-Meshes
            for i, mesh_data in enumerate(inst['meshes']):
                # 构造文件名: instanceID_subIndex.obj
                mesh_name = f"{uid}_m{i}"
                filename = f"{mesh_name}.obj"

                # 写入磁盘
                self._write_obj_file(filename, mesh_data['vertices'], mesh_data['faces'])

                # 在 MuJoCo Spec 中添加 Asset
                # 注意：MuJoCo 会根据 file 扩展名自动识别类型
                mesh_asset = self.spec.add_mesh()
                mesh_asset.name = mesh_name
                mesh_asset.file = os.path.join(self.asset_dir, filename)

    def _get_joint_enum(self, raw_type: str):
        t = str(raw_type).lower()
        if "slider" in t or "2" in t:
            return mujoco.mjtJoint.mjJNT_SLIDE
        if "hinge" in t:
            return mujoco.mjtJoint.mjJNT_HINGE
        return None

    def _get_local_pose(self, parent_abs_pos, parent_abs_quat, child_abs_pos, child_abs_quat):
        """
        计算子物体相对于父物体的局部变换 (Pos, Quat)
        """
        # 1. 计算父物体旋转的逆 (共轭四元数: w, -x, -y, -z)
        inv_parent_quat = np.array([parent_abs_quat[0], -parent_abs_quat[1], -parent_abs_quat[2], -parent_abs_quat[3]])

        # 2. 计算相对位置: R_parent_inv * (P_child - P_parent)
        diff_pos = child_abs_pos - parent_abs_pos
        local_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(local_pos, diff_pos, inv_parent_quat)

        # 3. 计算相对旋转: R_parent_inv * R_child
        local_quat = np.zeros(4)
        mujoco.mju_mulQuat(local_quat, inv_parent_quat, child_abs_quat)

        return local_pos, local_quat

    def _add_body_recursive(self, parent_body, node: Dict, parent_abs_pos, parent_abs_quat):
        body_id = node["body_id"]
        link_data = node.get("link_data")
        self.processed_ids.add(body_id)

        # 1. 创建 Body (容器)
        # -------------------------------------------------
        body_pos = np.array([0.0, 0.0, 0.0])
        body_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # 从 Instance 获取 Body 的位姿 (假设是相对坐标，且为弧度)
        if body_id in self.instance_map:
            inst = self.instance_map[body_id]
            body_pos = np.array(inst['position'])
            # 弧度转四元数
            if 'rotation' in inst:
                mujoco.mju_euler2Quat(body_quat, np.array(inst['rotation']), "XYZ")
        rel_pos, rel_quat = self._get_local_pose(parent_abs_pos, parent_abs_quat, body_pos, body_quat)
        current_body = parent_body.add_body()
        current_body.name = body_id
        current_body.pos = rel_pos
        current_body.quat = rel_quat
        current_body.mass = 1.0

        # 2. 添加 Geoms (支持一个 Body 包含多个 Meshes)
        # -------------------------------------------------
        if body_id in self.instance_map:
            inst = self.instance_map[body_id]
            # 这里的 index 'i' 必须与 _prepare_assets 中生成 obj 文件时的命名逻辑一致
            for i in range(len(inst['meshes'])):
                geom = current_body.add_geom()
                geom.type = mujoco.mjtGeom.mjGEOM_MESH
                # 对应 _prepare_assets 生成的资源名: "ID_m0", "ID_m1"...
                geom.meshname = f"{body_id}_m{i}"
                geom.pos = np.zeros(3)  # 默认 mesh 原点与 body 重合
                # 可选: 设置不同 mesh 的颜色或材质
                geom.rgba = np.array([0.8, 0.8, 0.8, 1.0])

                # 3. 设置 Joint & Actuator (来自 Auxiliary)
        # -------------------------------------------------
        if link_data:
            j_type = self._get_joint_enum(link_data.get("joint_type", ""))
            if j_type is not None:
                joint = current_body.add_joint()
                joint.name = f"joint_{body_id}"
                joint.type = j_type

                if "position" in link_data:
                    joint_pos = np.array(link_data["position"])
                    joint_local_pos, _ = self._get_local_pose(body_pos, body_quat, joint_pos,
                                                              np.array([1., 0., 0., 0.]))
                    joint.pos = joint_local_pos

                if "rotation" in link_data:
                    joint.axis = np.array(link_data["rotation"])
                else:
                    joint.axis = np.array([0, 0, 1])

                if link_data.get("hinge_actuator"):
                    act = self.spec.add_actuator()
                    act.name = f"actuator_{body_id}"
                    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                    act.target = joint.name

        # 4. 递归子节点
        for child in node.get("children", []):
            self._add_body_recursive(current_body, child, body_pos, body_quat)

    def build(self, topology_forest: List[Dict], instances: List[Dict]):
        """
        主构建函数
        """
        # 1. 准备资源 (生成 OBJ 并注册 Mesh 资源)
        self._prepare_assets(instances)

        # 2. 构建拓扑树 (处理有父子关系的 Body)
        # 根节点直接挂在 Worldbody 下
        world_pos = np.array([0.0, 0.0, 0.0])
        world_quat = np.array([1.0, 0.0, 0.0, 0.0])
        for root_node in topology_forest:
            self._add_body_recursive(self.spec.worldbody, root_node,world_pos,world_quat)

        # 3. 处理孤立物体 (Orphans / Fixed Bodies)
        # -------------------------------------------------
        for inst in instances:
            uid = inst['unique_id']

            # 如果该 ID 没有在拓扑树中处理过，则作为孤立物体处理
            if uid not in self.processed_ids:
                # A. 创建 Body
                body = self.spec.worldbody.add_body()
                body.name = uid
                body.pos = np.array(inst['position'])
                body.mass = 1.0

                quat = np.zeros(4)
                if 'rotation' in inst:
                    mujoco.mju_euler2Quat(quat, np.array(inst['rotation']), "XYZ")
                else:
                    quat = np.array([1., 0., 0., 0.])
                body.quat = quat

                # B. 循环添加 Geoms
                # 即使是孤立物体，也可能由多个 sub-meshes 组成
                for i in range(len(inst['meshes'])):
                    geom = body.add_geom()
                    geom.type = mujoco.mjtGeom.mjGEOM_MESH
                    geom.meshname = f"{uid}_m{i}"
                    geom.rgba = np.array([0.5, 0.5, 1.0, 1.0])  # 蓝色区分孤立物体

    def save_xml(self, filename: str):
        xml_str = self.spec.to_xml()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        return xml_str


# --- 测试数据与执行 ---

# 1. 所有的 Body 列表
all_bodies = parse_editor3d_json.parse_editor3d_json("editor3dJson.json")

# 2. 拓扑树结构
auxiliaries=normalize_auxiliaries("editor3dJson.json")
topology_tree = build_topology_tree(auxiliaries)

# 3. 运行构建
try:
    builder = MujocoBuilderWithMesh("editor_data_mj_spec", asset_dir="my_assets")
    builder.build(topology_tree, all_bodies)

    # 4. 获取 XML
    xml_content = builder.save_xml('test.xml')

    print(xml_content)

    # 5. 验证是否可以被编译 (可选)
    # model = builder.spec.compile()
    # print("Model compiled successfully!")

except AttributeError as e:
    print("错误: 你的 mujoco 版本可能过低，不支持 MjSpec。")
    print("请运行: pip install --upgrade mujoco")
    print(f"详细错误: {e}")






